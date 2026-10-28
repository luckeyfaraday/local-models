import Foundation
import llama

struct ModelInfo: Equatable {
    let name: String
    let architecture: String
    let parameterCount: UInt64
    let sizeBytes: UInt64
    let trainContext: Int
    let contextLength: Int
    let chatTemplate: String?
    let addsBOS: Bool
}

struct Sampling: Equatable {
    var temperature: Float
    var topK: Int32
    var topP: Float
    var repeatPenalty: Float
}

struct GenerationStats: Equatable {
    var promptTokens = 0
    var promptSeconds = 0.0
    var generatedTokens = 0
    var generationSeconds = 0.0
    var hitContextLimit = false

    var tokensPerSecond: Double { generationSeconds > 0 ? Double(generatedTokens) / generationSeconds : 0 }
    var promptTokensPerSecond: Double { promptSeconds > 0 ? Double(promptTokens) / promptSeconds : 0 }
}

enum EngineError: LocalizedError {
    case loadFailed(String)
    case contextFailed
    case notLoaded
    case promptTooLong(tokens: Int, context: Int)
    case decodeFailed(Int32)

    var errorDescription: String? {
        switch self {
        case .loadFailed(let file):
            return "Couldn't load \(file). The file may be incomplete, or this llama.cpp build doesn't support its architecture."
        case .contextFailed:
            return "Couldn't create a context for this model. Try a smaller context length in Settings."
        case .notLoaded:
            return "No model is loaded."
        case .promptTooLong(let tokens, let context):
            return "This chat is \(tokens) tokens, more than the \(context)-token context. Start a new chat or raise the context length in Settings."
        case .decodeFailed(let code):
            return "llama_decode failed with code \(code)."
        }
    }
}

/// One llama.cpp model and context. Every llama call runs on `queue`, so the
/// C state is only ever touched from one thread at a time.
final class LlamaEngine: @unchecked Sendable {
    enum Event {
        case text(String)
        case done(GenerationStats)
    }

    private static let backendReady: Void = llama_backend_init()

    private let queue = DispatchQueue(label: "llama.engine", qos: .userInitiated)
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var nCtx = 0

    private let cancelLock = NSLock()
    private var cancelled = false

    init() {
        _ = LlamaEngine.backendReady
    }

    deinit {
        unloadNow()
    }

    // MARK: - Public API

    func load(path: String, contextLength: Int) async throws -> ModelInfo {
        try await onQueue { try self.loadNow(path: path, contextLength: contextLength) }
    }

    func unload() async {
        try? await onQueue { self.unloadNow() }
    }

    func cancel() {
        cancelLock.lock()
        cancelled = true
        cancelLock.unlock()
    }

    func generate(prompt: String, addBOS: Bool, sampling: Sampling, maxTokens: Int) -> AsyncThrowingStream<Event, Error> {
        setCancelled(false)
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }
            queue.async {
                do {
                    let stats = try self.generateNow(prompt: prompt, addBOS: addBOS, sampling: sampling, maxTokens: maxTokens) {
                        continuation.yield(.text($0))
                    }
                    continuation.yield(.done(stats))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Work on the engine queue

    private func onQueue<T>(_ body: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do { continuation.resume(returning: try body()) } catch { continuation.resume(throwing: error) }
            }
        }
    }

    private var isCancelled: Bool {
        cancelLock.lock()
        defer { cancelLock.unlock() }
        return cancelled
    }

    private func setCancelled(_ value: Bool) {
        cancelLock.lock()
        cancelled = value
        cancelLock.unlock()
    }

    private func loadNow(path: String, contextLength: Int) throws -> ModelInfo {
        unloadNow()

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = -1 // everything on Metal
        guard let loaded = llama_model_load_from_file(path, modelParams) else {
            throw EngineError.loadFailed((path as NSString).lastPathComponent)
        }

        let trainContext = Int(llama_model_n_ctx_train(loaded))
        let wanted = trainContext > 0 ? min(contextLength, trainContext) : contextLength
        let threads = Int32(max(1, min(4, ProcessInfo.processInfo.activeProcessorCount - 2)))

        var contextParams = llama_context_default_params()
        contextParams.n_ctx = UInt32(wanted)
        contextParams.n_batch = 512
        contextParams.n_ubatch = 512
        contextParams.n_threads = threads
        contextParams.n_threads_batch = threads
        guard let ctx = llama_init_from_model(loaded, contextParams) else {
            llama_model_free(loaded)
            throw EngineError.contextFailed
        }

        model = loaded
        context = ctx
        vocab = llama_model_get_vocab(loaded)
        nCtx = Int(llama_n_ctx(ctx))

        var template: String?
        if let raw = llama_model_chat_template(loaded, nil) {
            template = String(cString: raw)
        }
        let fallbackName = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        return ModelInfo(
            name: meta(loaded, "general.name") ?? fallbackName,
            architecture: meta(loaded, "general.architecture") ?? "unknown",
            parameterCount: llama_model_n_params(loaded),
            sizeBytes: llama_model_size(loaded),
            trainContext: trainContext,
            contextLength: nCtx,
            chatTemplate: template,
            addsBOS: llama_vocab_get_add_bos(vocab)
        )
    }

    private func unloadNow() {
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        context = nil
        model = nil
        vocab = nil
        nCtx = 0
    }

    private func generateNow(prompt: String, addBOS: Bool, sampling: Sampling, maxTokens: Int, emit: (String) -> Void) throws -> GenerationStats {
        guard let ctx = context, let vocab else { throw EngineError.notLoaded }

        var tokens = tokenize(prompt, vocab: vocab)
        if addBOS {
            let bos = llama_vocab_bos(vocab)
            if bos >= 0 && tokens.first != bos { tokens.insert(bos, at: 0) }
        }
        if tokens.count >= nCtx - 8 { throw EngineError.promptTooLong(tokens: tokens.count, context: nCtx) }

        // Hybrid models (LFM's conv layers) can't drop part of their state, so
        // every turn starts from a clean memory and re-reads the whole chat.
        llama_memory_clear(llama_get_memory(ctx), true)

        let sampler = makeSampler(sampling, vocab: vocab)
        defer { llama_sampler_free(sampler) }

        var stats = GenerationStats()
        stats.promptTokens = tokens.count

        let promptStart = DispatchTime.now().uptimeNanoseconds
        let batchSize = Int(llama_n_batch(ctx))
        var offset = 0
        while offset < tokens.count {
            if isCancelled { return stats }
            let count = min(batchSize, tokens.count - offset)
            let start = offset
            let rc = tokens.withUnsafeMutableBufferPointer { buffer in
                llama_decode(ctx, llama_batch_get_one(buffer.baseAddress! + start, Int32(count)))
            }
            if rc != 0 { throw EngineError.decodeFailed(rc) }
            offset += count
        }
        let generationStart = DispatchTime.now().uptimeNanoseconds
        stats.promptSeconds = Double(generationStart - promptStart) / 1e9

        var pending: [UInt8] = []
        var position = tokens.count
        while stats.generatedTokens < maxTokens {
            if isCancelled { break }
            if position >= nCtx {
                stats.hitContextLimit = true
                break
            }
            var token = llama_sampler_sample(sampler, ctx, -1)
            if llama_vocab_is_eog(vocab, token) { break }

            pending += piece(token, vocab: vocab)
            if let text = String(bytes: pending, encoding: .utf8) {
                emit(text)
                pending.removeAll()
            } else if pending.count > 16 {
                emit(String(decoding: pending, as: UTF8.self))
                pending.removeAll()
            }

            let rc = withUnsafeMutablePointer(to: &token) { pointer in
                llama_decode(ctx, llama_batch_get_one(pointer, 1))
            }
            if rc != 0 { throw EngineError.decodeFailed(rc) }
            position += 1
            stats.generatedTokens += 1
        }
        if !pending.isEmpty { emit(String(decoding: pending, as: UTF8.self)) }
        stats.generationSeconds = Double(DispatchTime.now().uptimeNanoseconds - generationStart) / 1e9
        return stats
    }

    // MARK: - Helpers

    private func makeSampler(_ s: Sampling, vocab: OpaquePointer) -> UnsafeMutablePointer<llama_sampler> {
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        if s.repeatPenalty != 1 {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, s.repeatPenalty, 0, 0))
        }
        if s.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
            return chain
        }
        if s.topK > 0 { llama_sampler_chain_add(chain, llama_sampler_init_top_k(s.topK)) }
        if s.topP < 1 { llama_sampler_chain_add(chain, llama_sampler_init_top_p(s.topP, 1)) }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(s.temperature))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: 0..<UInt32.max)))
        return chain
    }

    private func tokenize(_ text: String, vocab: OpaquePointer) -> [llama_token] {
        let byteCount = Int32(text.utf8.count)
        var capacity = Int(byteCount) + 8
        var tokens = [llama_token](repeating: 0, count: capacity)
        var n = llama_tokenize(vocab, text, byteCount, &tokens, Int32(capacity), false, true)
        if n < 0 {
            capacity = Int(-n)
            tokens = [llama_token](repeating: 0, count: capacity)
            n = llama_tokenize(vocab, text, byteCount, &tokens, Int32(capacity), false, true)
        }
        return Array(tokens.prefix(Int(max(n, 0))))
    }

    /// Bytes for one token. Special tokens are rendered so `<think>` tags show up.
    private func piece(_ token: llama_token, vocab: OpaquePointer) -> [UInt8] {
        var capacity = 32
        var buffer = [CChar](repeating: 0, count: capacity)
        var n = llama_token_to_piece(vocab, token, &buffer, Int32(capacity), 0, true)
        if n < 0 {
            capacity = Int(-n)
            buffer = [CChar](repeating: 0, count: capacity)
            n = llama_token_to_piece(vocab, token, &buffer, Int32(capacity), 0, true)
        }
        return buffer.prefix(Int(max(n, 0))).map { UInt8(bitPattern: $0) }
    }

    private func meta(_ model: OpaquePointer, _ key: String) -> String? {
        let capacity = 512
        var buffer = [CChar](repeating: 0, count: capacity)
        guard llama_model_meta_val_str(model, key, &buffer, capacity) >= 0 else { return nil }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        let value = String(decoding: bytes, as: UTF8.self)
        return value.isEmpty ? nil : value
    }
}

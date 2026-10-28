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
    /// The mmproj file loaded next to the model, if any.
    var projector: String?
    var hearsAudio = false
    var seesImages = false
    var speaks = false

    /// What the projector adds to text, e.g. "audio in, speech out".
    var extras: String {
        let parts = [seesImages ? "images in" : nil, hearsAudio ? "audio in" : nil, speaks ? "speech out" : nil]
        let list = parts.compactMap { $0 }
        return list.isEmpty ? "nothing extra" : list.joined(separator: ", ")
    }
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
    case projectorFailed(String)
    case cannotSpeak
    case speechFailed(String)

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
        case .projectorFailed(let file):
            return "Couldn't load the projector \(file). It may be incomplete or belong to another model."
        case .cannotSpeak:
            return "The loaded model can't generate speech."
        case .speechFailed(let step):
            return "Speech generation failed: \(step)."
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
    private var mtmd: OpaquePointer?
    private var audioGen: OpaquePointer?
    private var speaks = false

    private let cancelLock = NSLock()
    private var cancelled = false

    init() {
        _ = LlamaEngine.backendReady
    }

    deinit {
        unloadNow()
    }

    // MARK: - Public API

    func load(path: String, projector: String?, contextLength: Int) async throws -> ModelInfo {
        try await onQueue { try self.loadNow(path: path, projector: projector, contextLength: contextLength) }
    }

    func unload() async {
        try? await onQueue { self.unloadNow() }
    }

    func cancel() {
        cancelLock.lock()
        cancelled = true
        cancelLock.unlock()
    }

    func generate(prompt: String, addBOS: Bool, sampling: Sampling, maxTokens: Int,
                  banned: [String] = []) -> AsyncThrowingStream<Event, Error> {
        setCancelled(false)
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }
            queue.async {
                do {
                    let stats = try self.generateNow(prompt: prompt, addBOS: addBOS, sampling: sampling,
                                                     maxTokens: maxTokens, banned: banned) {
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

    /// Text to speech. Audio arrives in pieces as it is generated, so playback
    /// can start before the whole sentence is done.
    func speak(_ request: SpeechRequest) -> AsyncThrowingStream<SpeechEvent, Error> {
        setCancelled(false)
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }
            queue.async {
                do {
                    let stats = try self.speakNow(request) { samples, rate in
                        continuation.yield(.audio(samples, sampleRate: rate))
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

    private func loadNow(path: String, projector: String?, contextLength: Int) throws -> ModelInfo {
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
        // Speech generation feeds the model's hidden state to the audio decoder.
        let projectorCaps = projector.map { mtmd_get_cap_from_file($0) }
        contextParams.embeddings = projectorCaps?.gen_audio ?? false
        guard let ctx = llama_init_from_model(loaded, contextParams) else {
            llama_model_free(loaded)
            throw EngineError.contextFailed
        }

        var media: OpaquePointer?
        if let projector {
            var mediaParams = mtmd_context_params_default()
            mediaParams.use_gpu = true
            mediaParams.n_threads = threads
            mediaParams.print_timings = false
            mediaParams.warmup = false
            media = mtmd_init_from_file(projector, loaded, mediaParams)
            if media == nil {
                llama_free(ctx)
                llama_model_free(loaded)
                throw EngineError.projectorFailed((projector as NSString).lastPathComponent)
            }
        }
        let caps = mtmd_helper_get_capabilities(loaded, media)

        model = loaded
        context = ctx
        vocab = llama_model_get_vocab(loaded)
        nCtx = Int(llama_n_ctx(ctx))
        mtmd = media
        speaks = caps.audio_gen && contextParams.embeddings

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
            addsBOS: llama_vocab_get_add_bos(vocab),
            projector: projector.map { ($0 as NSString).lastPathComponent },
            hearsAudio: caps.audio,
            seesImages: caps.vision,
            speaks: speaks
        )
    }

    private func unloadNow() {
        if let audioGen { mtmd_helper_gen_audio_free(audioGen) }
        if let mtmd { mtmd_free(mtmd) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        audioGen = nil
        mtmd = nil
        context = nil
        model = nil
        vocab = nil
        nCtx = 0
        speaks = false
    }

    private func generateNow(prompt: String, addBOS: Bool, sampling: Sampling, maxTokens: Int, banned: [String],
                             emit: (String) -> Void) throws -> GenerationStats {
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

        // Only single-token strings (special tags like <think>) can be banned.
        // A speaking model stays in text here: it must not switch to audio output.
        let banned = speaks ? banned + ["<|audio_start|>", "<|mixed_start|>"] : banned
        let bannedTokens = banned.compactMap { text -> llama_token? in
            let ids = tokenize(text, vocab: vocab)
            return ids.count == 1 ? ids[0] : nil
        }
        let sampler = makeSampler(sampling, vocab: vocab, banned: bannedTokens)
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

    private func speakNow(_ request: SpeechRequest, emit: @escaping ([Float], Int) -> Void) throws -> SpeechStats {
        guard let ctx = context, let vocab, let mtmd else { throw EngineError.notLoaded }
        guard speaks else { throw EngineError.cannotSpeak }
        if audioGen == nil { audioGen = mtmd_helper_gen_audio_init(ctx, mtmd) }
        guard let gen = audioGen else { throw EngineError.speechFailed("the audio decoder didn't start") }

        llama_memory_clear(llama_get_memory(ctx), true)

        let type = mtmd_gen_audio_get_info(mtmd).type
        // LFM3-Audio is told what to say through its "tts" chat template; other
        // speech models (Qwen3-TTS, Soprano...) take the plain text.
        let prompt: String
        if type == MTMD_GEN_AUDIO_TYPE_LFM3A {
            let bos = String(cString: llama_vocab_get_text(vocab, llama_vocab_bos(vocab)))
            prompt = SpeechRequest.lfm3Prompt(request, bos: bos)
        } else {
            prompt = request.text
        }

        var stats = SpeechStats()
        stats.promptTokens = tokenize(prompt, vocab: vocab).count
        let start = DispatchTime.now().uptimeNanoseconds
        func seconds(since t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9 }

        // The C strings have to stay alive while the helper works.
        let promptC = strdup(prompt), langC = strdup(request.language), voiceC = strdup(request.voice)
        defer { free(promptC); free(langC); free(voiceC) }
        var input = mtmd_helper_gen_audio_inp_default()
        input.stream = true
        input.seq_id = 0
        input.prompt = UnsafePointer(promptC)
        input.prompt_len = strlen(promptC!)
        input.n_past = 0
        input.lang = UnsafePointer(langC)
        input.voice = UnsafePointer(voiceC)
        input.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM
        guard mtmd_helper_gen_audio_set_input(gen, &input) == 0 else {
            throw EngineError.speechFailed("the model rejected the prompt")
        }

        let batch = Int32(llama_n_batch(ctx))
        while true {
            if isCancelled { stats.stopped = true; return stats }
            let left = mtmd_helper_gen_audio_step_prompt(gen, batch)
            if left < 0 { throw EngineError.speechFailed("reading the text") }
            if left == 0 { break }
        }
        let generationStart = DispatchTime.now().uptimeNanoseconds
        stats.promptSeconds = Double(generationStart - start) / 1e9

        // Audio produced since the last call. In stream mode it has to be read after every frame.
        func drain() throws {
            var rate: Int32 = 0
            var data: UnsafePointer<CChar>?
            var length = 0
            var count: Int64 = 0
            guard mtmd_helper_gen_audio_get_output(gen, &rate, &data, &length, &count) == 0 else {
                throw EngineError.speechFailed("decoding audio")
            }
            stats.sampleRate = Int(rate)
            let n = length / MemoryLayout<Float>.size
            guard n > 0, let data else { return }
            let samples = [Float](unsafeUninitializedCapacity: n) { buffer, initialized in
                memcpy(buffer.baseAddress!, data, n * MemoryLayout<Float>.size)
                initialized = n
            }
            if stats.firstAudioSeconds == nil { stats.firstAudioSeconds = seconds(since: start) }
            stats.samples += n
            emit(samples, Int(rate))
        }

        // Pipelines with a discrete backbone token (Qwen3-TTS) need one sampled
        // every frame; LFM3-Audio speaks from the hidden state alone.
        let sampler = makeSampler(Sampling(temperature: 0.8, topK: 40, topP: 0.95, repeatPenalty: 1), vocab: vocab, banned: [])
        defer { llama_sampler_free(sampler) }
        let needsToken = type != MTMD_GEN_AUDIO_TYPE_LFM3A

        let maxFrames = max(0, min(SpeechRequest.maxFrames, nCtx - stats.promptTokens - 16))
        guard let promptHidden = llama_get_embeddings_ith(ctx, -1) else {
            throw EngineError.speechFailed("no hidden state after the prompt")
        }
        var hidden: UnsafePointer<Float>? = UnsafePointer(promptHidden)
        while stats.frames < maxFrames {
            if isCancelled { stats.stopped = true; break }
            var token: llama_token = -1
            if needsToken || mtmd_helper_gen_audio_next_is_text(gen) {
                token = llama_sampler_sample(sampler, ctx, -1)
            }
            var next: UnsafePointer<Float>?
            var stop = false
            guard mtmd_helper_gen_audio_step_gen(gen, token, hidden, &next, &stop) == 0 else {
                throw EngineError.speechFailed("frame \(stats.frames + 1)")
            }
            try drain()
            if stop || next == nil { break }
            stats.frames += 1
            hidden = next
        }
        stats.hitFrameLimit = stats.frames >= maxFrames
        guard mtmd_helper_gen_audio_flush(gen) == 0 else { throw EngineError.speechFailed("the last frame") }
        try drain()
        stats.generationSeconds = seconds(since: generationStart)
        return stats
    }

    // MARK: - Helpers

    private func makeSampler(_ s: Sampling, vocab: OpaquePointer, banned: [llama_token]) -> UnsafeMutablePointer<llama_sampler> {
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        if !banned.isEmpty {
            let biases = banned.map { llama_logit_bias(token: $0, bias: -Float.infinity) }
            llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), Int32(biases.count), biases))
        }
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

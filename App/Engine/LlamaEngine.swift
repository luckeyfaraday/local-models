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
    /// Spoken conversation: audio in, interleaved text and speech out (LFM3-Audio).
    var talks = false
    /// Sample rates of the audio the projector hears and speaks, in Hz.
    var audioInputRate = 16000
    var audioOutputRate = 24000

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

enum EngineError: LocalizedError {
    case loadFailed(String)
    case contextFailed
    case notLoaded
    case promptTooLong(tokens: Int, context: Int)
    case decodeFailed(Int32)
    case projectorFailed(String)
    case cannotSpeak
    case speechFailed(String)
    case cannotTalk
    case voiceFailed(String)
    case cannotSee
    case imageFailed(String)

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
        case .cannotTalk:
            return "The loaded model can't hold a spoken conversation."
        case .voiceFailed(let step):
            return "The voice reply failed: \(step)."
        case .cannotSee:
            return "The loaded model can't see images. Load a vision model with its projector, like LFM3-VL-2B."
        case .imageFailed(let step):
            return "Reading the image failed: \(step)."
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

    /// Most tokens one image may take. LFM3-VL picks its resolution per image
    /// (32×32 pixels per token, up to 4096 tokens); 1024 keeps a photo or a
    /// screenshot readable while the encoder and prefill stay quick on a phone.
    static let imageMaxTokens: Int32 = 1024

    private let queue = DispatchQueue(label: "llama.engine", qos: .userInitiated)
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var nCtx = 0
    private var mtmd: OpaquePointer?
    private var audioGen: OpaquePointer?
    private var speaks = false
    private var talks = false

    /// What the memory holds after a chat turn: the prompt and the reply as
    /// UTF-8 text (without the BOS), the images in it, and where it ends.
    private struct ChatCache {
        var bytes: [UInt8]
        var images: [Data]
        var addBOS: Bool
        var tokens: Int
        var position: llama_pos
    }

    /// One chunk of an image, as the vision encoder sees it.
    private struct ImageKey: Equatable {
        let image: Data
        let part: Int
        let tokens: Int
    }

    private var chatCache: ChatCache?
    /// Embeddings of the images read lately, oldest first. An image of 1024
    /// tokens is about 8 MB of them.
    private var imageCache: [(key: ImageKey, rows: [Float])] = []
    static let imageCacheBytes = 48 << 20

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

    /// Memory is short: the kept image embeddings can be made again.
    func dropImageCache() {
        queue.async { self.imageCache.removeAll() }
    }

    func cancel() {
        cancelLock.lock()
        cancelled = true
        cancelLock.unlock()
    }

    /// Streams a reply. `prompt` holds one media marker (`mtmd_default_marker()`)
    /// per entry of `images` (encoded image files), in the same order.
    func generate(prompt: String, addBOS: Bool, images: [Data] = [], sampling: Sampling, maxTokens: Int,
                  banned: [String] = []) -> AsyncThrowingStream<Event, Error> {
        setCancelled(false)
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }
            queue.async {
                do {
                    let stats = try self.generateNow(prompt: prompt, addBOS: addBOS, images: images, sampling: sampling,
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
                    var stats: SpeechStats?
                    let chunks = request.chunks
                    for chunk in chunks {
                        var part = request
                        part.text = chunk
                        let partStats = try self.speakNow(part) { samples, rate in
                            continuation.yield(.audio(samples, sampleRate: rate))
                        }
                        if stats == nil { stats = partStats } else { stats!.append(partStats) }
                        if partStats.stopped || self.isCancelled { break }
                    }
                    stats?.chunks = chunks.count
                    continuation.yield(.done(stats ?? SpeechStats()))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// One turn of a spoken conversation: the user's audio (mono PCM at the
    /// projector's rate) after the earlier turns, answered with text and speech
    /// interleaved in one generation. Both stream out as they are made.
    func voiceReply(history: [VoiceExchangeText], audio: [Float]) -> AsyncThrowingStream<VoiceEvent, Error> {
        setCancelled(false)
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] reason in
                if case .cancelled = reason { self?.cancel() }
            }
            queue.async {
                do {
                    let prefix = VoicePrompt.chat(history: history, bos: self.bosText())
                    let stats = try self.respondNow(prefix: prefix, audio: audio, mode: MTMD_HELPER_GEN_AUDIO_OUT_MIXED,
                                                    maxSteps: VoicePrompt.maxSteps,
                                                    onText: { continuation.yield(.text($0)) },
                                                    onAudio: { continuation.yield(.audio($0, sampleRate: $1)) })
                    continuation.yield(.done(stats))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// The words of a clip, transcribed by the same model.
    func transcribe(_ audio: [Float], language: String = "en") async throws -> String {
        setCancelled(false)
        return try await onQueue { () throws -> String in
            var text = ""
            _ = try self.respondNow(prefix: VoicePrompt.asr(language: language, bos: self.bosText()), audio: audio,
                                    mode: MTMD_HELPER_GEN_AUDIO_OUT_TEXT, maxSteps: 400,
                                    onText: { text += $0 }, onAudio: { _, _ in })
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
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
            mediaParams.image_max_tokens = LlamaEngine.imageMaxTokens
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
        // Voice chat mirrors llama-server's /v1/realtime: audio goes in through
        // the audio stream, the reply comes from the LFM3-Audio gen pipeline.
        talks = speaks && caps.audio && media.map {
            mtmd_support_audio_stream($0) && mtmd_gen_audio_get_info($0).type == MTMD_GEN_AUDIO_TYPE_LFM3A
        } == true
        let audioInputRate = media.map { Int(mtmd_get_audio_sample_rate($0)) } ?? 0
        let audioOutputRate = speaks ? media.map { Int(mtmd_gen_audio_get_info($0).sample_rate) } ?? 0 : 0

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
            speaks: speaks,
            talks: talks,
            audioInputRate: audioInputRate > 0 ? audioInputRate : 16000,
            audioOutputRate: audioOutputRate > 0 ? audioOutputRate : 24000
        )
    }

    private func unloadNow() {
        if let audioGen { mtmd_helper_gen_audio_free(audioGen) }
        if let mtmd { mtmd_free(mtmd) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        chatCache = nil
        imageCache.removeAll()
        audioGen = nil
        mtmd = nil
        context = nil
        model = nil
        vocab = nil
        nCtx = 0
        speaks = false
        talks = false
    }

    private func generateNow(prompt: String, addBOS: Bool, images: [Data], sampling: Sampling, maxTokens: Int,
                             banned: [String], emit: (String) -> Void) throws -> GenerationStats {
        guard let ctx = context, let vocab, let model else { throw EngineError.notLoaded }

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
        stats.images = images.count
        let promptStart = DispatchTime.now().uptimeNanoseconds

        // The memory still holds the last turn. If this prompt is that turn
        // with more after it, only the new part is read; anything else starts
        // from a clean memory, since hybrid models (LFM's conv layers) can't
        // drop part of their state.
        let promptBytes = Array(prompt.utf8)
        let cache = chatCache
        chatCache = nil // until this turn is done, the memory matches nothing
        var start = (tokens: 0, position: llama_pos(0))
        var text = prompt
        var newImages = images
        if let cache, cache.addBOS == addBOS, images.count >= cache.images.count,
           Array(images.prefix(cache.images.count)) == cache.images,
           let at = PromptCache.newPartStart(of: promptBytes, after: cache.bytes),
           memoryHolds(cache, ctx: ctx) {
            start = (cache.tokens, cache.position)
            text = String(decoding: promptBytes[at...], as: UTF8.self)
            newImages = Array(images.dropFirst(cache.images.count))
            stats.reusedTokens = cache.tokens
        } else {
            llama_memory_clear(llama_get_memory(ctx), true)
        }
        let next = try read(text, images: newImages, addBOS: addBOS, after: start, ctx: ctx, vocab: vocab,
                            model: model, stats: &stats)
        guard var position = next else { return stats } // cancelled while reading
        let generationStart = DispatchTime.now().uptimeNanoseconds
        stats.promptSeconds = Double(generationStart - promptStart) / 1e9

        // Generated tokens go in at explicit positions, which stay right after
        // images that take fewer positions than tokens (M-RoPE models).
        var batch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(batch) }

        var pending: [UInt8] = []
        var reply: [UInt8] = []
        while stats.generatedTokens < maxTokens {
            if isCancelled { break }
            if stats.promptTokens + stats.generatedTokens >= nCtx {
                stats.hitContextLimit = true
                break
            }
            let token = llama_sampler_sample(sampler, ctx, -1)
            if llama_vocab_is_eog(vocab, token) { break }

            let bytes = piece(token, vocab: vocab)
            reply += bytes
            pending += bytes
            if let text = String(bytes: pending, encoding: .utf8) {
                emit(text)
                pending.removeAll()
            } else if pending.count > 16 {
                emit(String(decoding: pending, as: UTF8.self))
                pending.removeAll()
            }

            batch.token[0] = token
            batch.pos[0] = position
            batch.n_seq_id[0] = 1
            batch.seq_id[0]![0] = 0
            batch.logits[0] = 1
            batch.n_tokens = 1
            let rc = llama_decode(ctx, batch)
            if rc != 0 { throw EngineError.decodeFailed(rc) }
            position += 1
            stats.generatedTokens += 1
        }
        if !pending.isEmpty { emit(String(decoding: pending, as: UTF8.self)) }
        stats.generationSeconds = Double(DispatchTime.now().uptimeNanoseconds - generationStart) / 1e9
        // Every token of the reply was decoded: the next turn can go on from here.
        chatCache = ChatCache(bytes: promptBytes + reply, images: images, addBOS: addBOS,
                              tokens: stats.promptTokens + stats.generatedTokens, position: position)
        AppLog.note(String(format: "chat: %d prompt tokens (%d kept, %d images) in %.2f s, %d generated at %.1f tok/s",
                           stats.promptTokens, stats.reusedTokens ?? 0, stats.images, stats.promptSeconds,
                           stats.generatedTokens, stats.tokensPerSecond))
        return stats
    }

    /// Whether the memory is still as `cache` left it. Without images the
    /// last position is known exactly; with them (M-RoPE counts differently)
    /// it only has to be there.
    private func memoryHolds(_ cache: ChatCache, ctx: OpaquePointer) -> Bool {
        let last = llama_memory_seq_pos_max(llama_get_memory(ctx), 0)
        return cache.images.isEmpty ? last == cache.position - 1 : last >= 0
    }

    /// Reads `text` into the context after what is already there (`start`),
    /// with one media marker per image. Returns the next position, or nil if cancelled.
    private func read(_ text: String, images: [Data], addBOS: Bool, after start: (tokens: Int, position: llama_pos),
                      ctx: OpaquePointer, vocab: OpaquePointer, model: OpaquePointer,
                      stats: inout GenerationStats) throws -> llama_pos? {
        let fresh = start.tokens == 0
        if images.isEmpty {
            var tokens = tokenize(text, vocab: vocab)
            if fresh && addBOS {
                let bos = llama_vocab_bos(vocab)
                if bos >= 0 && tokens.first != bos { tokens.insert(bos, at: 0) }
            }
            let total = start.tokens + tokens.count
            if total >= nCtx - 8 { throw EngineError.promptTooLong(tokens: total, context: nCtx) }
            stats.promptTokens = total

            // Explicit positions, so the text lands right after an image that
            // took fewer positions than tokens.
            var position = start.position
            let batchSize = Int(llama_n_batch(ctx))
            var offset = 0
            while offset < tokens.count {
                if isCancelled { return nil }
                let count = min(batchSize, tokens.count - offset)
                try tokens.withUnsafeBufferPointer { buffer in
                    try decode(ctx, model: model, tokens: buffer.baseAddress! + offset, rows: nil, count: Int32(count),
                               at: &position, logitsLast: offset + count == tokens.count)
                }
                offset += count
            }
            return position
        }
        return try readWithImages(text, addBOS: fresh && addBOS, images: images, after: start, ctx: ctx, model: model,
                                  stats: &stats)
    }

    /// Reads a prompt with images in it, as llama-mtmd-cli does: mtmd splits it
    /// into text and image chunks at the media markers, text is decoded as
    /// tokens, each image goes through the vision encoder and its embeddings
    /// are decoded in place. The embeddings are kept, so an image that is read
    /// again skips the encoder. Returns the next position, or nil if cancelled.
    private func readWithImages(_ prompt: String, addBOS: Bool, images: [Data],
                                after start: (tokens: Int, position: llama_pos), ctx: OpaquePointer,
                                model: OpaquePointer, stats: inout GenerationStats) throws -> llama_pos? {
        guard let mtmd, mtmd_support_vision(mtmd) else { throw EngineError.cannotSee }

        var bitmaps: [OpaquePointer?] = []
        defer { bitmaps.forEach { mtmd_bitmap_free($0) } }
        for (i, data) in images.enumerated() {
            guard let image = RGBImage(data: data) else {
                throw EngineError.imageFailed("image \(i + 1) couldn't be decoded")
            }
            let bitmap = image.pixels.withUnsafeBufferPointer {
                mtmd_bitmap_init(UInt32(image.width), UInt32(image.height), $0.baseAddress)
            }
            guard let bitmap else { throw EngineError.imageFailed("image \(i + 1) is too large") }
            // The id comes back on the image's chunks: it says which image they belong to.
            mtmd_bitmap_set_id(bitmap, String(i))
            bitmaps.append(bitmap)
        }

        guard let chunks = mtmd_input_chunks_init() else { throw EngineError.imageFailed("out of memory") }
        defer { mtmd_input_chunks_free(chunks) }
        // The BOS goes in as text, as the text-only path inserts the token.
        let bos = bosText()
        let text = addBOS && !bos.isEmpty && !prompt.hasPrefix(bos) ? bos + prompt : prompt
        let rc = text.withCString { cText -> Int32 in
            var input = mtmd_input_text(text: cText, text_len: strlen(cText), add_special: false, parse_special: true)
            return bitmaps.withUnsafeBufferPointer { mtmd_tokenize(mtmd, chunks, &input, $0.baseAddress, $0.count) }
        }
        switch rc {
        case 0: break
        case 1: throw EngineError.imageFailed("the prompt and the images don't match")
        default: throw EngineError.imageFailed("the model couldn't prepare the image (\(rc))")
        }

        let tokens = start.tokens + mtmd_helper_get_n_tokens(chunks)
        if tokens >= nCtx - 8 { throw EngineError.promptTooLong(tokens: tokens, context: nCtx) }
        stats.promptTokens = tokens

        let batchSize = Int32(llama_n_batch(ctx))
        let count = mtmd_input_chunks_size(chunks)
        var nPast = start.position
        // An image can come as several chunks: how many of each image were seen.
        var parts: [Int: Int] = [:]
        for i in 0..<count {
            if isCancelled { return nil }
            let chunk = mtmd_input_chunks_get(chunks, i)
            let isImage = mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE
            // Text chunks add their token count to `next`, image chunks set it.
            var next = nPast
            var rc: Int32
            if isImage, let id = mtmd_input_chunk_get_id(chunk), let index = Int(String(cString: id)),
               images.indices.contains(index) {
                let part = parts[index, default: 0]
                parts[index] = part + 1
                let key = ImageKey(image: images[index], part: part, tokens: mtmd_input_chunk_get_n_tokens(chunk))
                let rows = try imageRows(for: key, chunk: chunk, mtmd: mtmd, model: model)
                // What mtmd_helper_eval_chunk_single does after its encoder pass. The rows are only read.
                rc = rows.withUnsafeBufferPointer {
                    mtmd_helper_decode_image_chunk(mtmd, ctx, chunk, UnsafeMutablePointer(mutating: $0.baseAddress), nPast, 0,
                                                   batchSize, &next, nil, nil)
                }
                if rc != 0 { throw EngineError.imageFailed("the model couldn't read the image (\(rc))") }
            } else {
                rc = mtmd_helper_eval_chunk_single(mtmd, ctx, chunk, nPast, 0, batchSize, i == count - 1, &next)
                if rc != 0 {
                    throw isImage ? EngineError.imageFailed("the vision encoder failed (\(rc))") : EngineError.decodeFailed(rc)
                }
            }
            nPast = next
        }
        return nPast
    }

    /// The embeddings of one image chunk: kept from an earlier turn, or from the vision encoder.
    private func imageRows(for key: ImageKey, chunk: OpaquePointer?, mtmd: OpaquePointer,
                           model: OpaquePointer) throws -> [Float] {
        if let i = imageCache.firstIndex(where: { $0.key == key }) {
            let entry = imageCache.remove(at: i)
            imageCache.append(entry)
            return entry.rows
        }
        let rc = mtmd_encode_chunk(mtmd, chunk)
        guard rc == 0, let output = mtmd_get_output_embd(mtmd) else {
            throw EngineError.imageFailed("the vision encoder failed (\(rc))")
        }
        let rows = Array(UnsafeBufferPointer(start: output, count: key.tokens * Int(llama_model_n_embd_inp(model))))
        imageCache.append((key, rows))
        // The oldest go first. The image just read always stays.
        var bytes = imageCache.reduce(0) { $0 + $1.rows.count * MemoryLayout<Float>.size }
        while bytes > LlamaEngine.imageCacheBytes && imageCache.count > 1 {
            bytes -= imageCache.removeFirst().rows.count * MemoryLayout<Float>.size
        }
        return rows
    }

    private func speakNow(_ request: SpeechRequest, emit: @escaping ([Float], Int) -> Void) throws -> SpeechStats {
        guard let ctx = context, let vocab, let mtmd else { throw EngineError.notLoaded }
        guard speaks else { throw EngineError.cannotSpeak }
        if audioGen == nil { audioGen = mtmd_helper_gen_audio_init(ctx, mtmd) }
        guard let gen = audioGen else { throw EngineError.speechFailed("the audio decoder didn't start") }

        chatCache = nil
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

    /// A turn the way llama-server's realtime slot runs it: the text before the
    /// audio as tokens, the audio as embedding rows from the audio stream, then
    /// the gen-audio pipeline continues the sequence in `mode` and follows the
    /// model's switching between text tokens and audio frames.
    private func respondNow(prefix: String, audio: [Float], mode: mtmd_helper_gen_audio_out_mode, maxSteps: Int,
                            onText: @escaping (String) -> Void,
                            onAudio: @escaping ([Float], Int) -> Void) throws -> VoiceStats {
        guard let ctx = context, let vocab, let mtmd, let model else { throw EngineError.notLoaded }
        guard talks else { throw EngineError.cannotTalk }
        if audioGen == nil { audioGen = mtmd_helper_gen_audio_init(ctx, mtmd) }
        guard let gen = audioGen else { throw EngineError.voiceFailed("the audio decoder didn't start") }

        var stats = VoiceStats()
        let start = DispatchTime.now().uptimeNanoseconds
        func seconds(since t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9 }

        // Every turn re-reads the conversation (earlier turns are text): the
        // hybrid memory can't drop part of itself, and transcription in between
        // uses the same sequence.
        chatCache = nil
        llama_memory_clear(llama_get_memory(ctx), true)
        var nPast: Int32 = 0
        let header = tokenize(prefix, vocab: vocab)
        try header.withUnsafeBufferPointer { buffer in
            try decode(ctx, model: model, tokens: buffer.baseAddress, rows: nil, count: Int32(buffer.count), at: &nPast)
        }

        guard let stream = mtmd_audio_stream_init(mtmd, mtmd_audio_stream_params_default()) else {
            throw EngineError.voiceFailed("the audio encoder didn't start")
        }
        defer { mtmd_audio_stream_free(stream) }
        _ = audio.withUnsafeBufferPointer { mtmd_audio_stream_push(stream, $0.baseAddress, $0.count) }
        stats.audioRows += try decodeAudioRows(stream, ctx: ctx, model: model, at: &nPast)
        _ = mtmd_audio_stream_flush(stream, true)
        stats.audioRows += try decodeAudioRows(stream, ctx: ctx, model: model, at: &nPast)

        // The C string has to stay alive while the helper reads the prompt.
        let openC = strdup(VoicePrompt.replyOpen)
        defer { free(openC) }
        var input = mtmd_helper_gen_audio_inp_default()
        input.stream = true
        input.seq_id = 0
        input.prompt = UnsafePointer(openC)
        input.prompt_len = strlen(openC!)
        input.n_past = nPast
        input.out_mode = mode
        input.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM
        guard mtmd_helper_gen_audio_set_input(gen, &input) == 0 else {
            throw EngineError.voiceFailed("the model rejected the conversation")
        }
        // The pipeline may bias text tokens, e.g. no switch to audio in a text-only reply.
        var nBias = 0
        var biases: [llama_logit_bias] = []
        if let bias = mtmd_helper_gen_audio_get_logit_bias(gen, &nBias), nBias > 0 {
            biases = Array(UnsafeBufferPointer(start: bias, count: nBias))
        }
        let batch = Int32(llama_n_batch(ctx))
        while true {
            if isCancelled { stats.stopped = true; return stats }
            let left = mtmd_helper_gen_audio_step_prompt(gen, batch)
            if left < 0 { throw EngineError.voiceFailed("reading the conversation") }
            if left == 0 { break }
        }
        stats.promptTokens = Int(nPast) + tokenize(VoicePrompt.replyOpen, vocab: vocab).count
        stats.promptSeconds = seconds(since: start)
        let generationStart = DispatchTime.now().uptimeNanoseconds

        // Text is greedy, as the GGUF says (top_k 1, temp 0); the pipeline
        // samples the audio codes with the model's own settings.
        let sampler = makeSampler(Sampling(temperature: 0, topK: 0, topP: 1, repeatPenalty: 1), vocab: vocab,
                                  banned: [], biases: biases)
        defer { llama_sampler_free(sampler) }

        let nEmbd = Int(llama_model_n_embd(model))
        guard let promptHidden = llama_get_embeddings_ith(ctx, -1) else {
            throw EngineError.voiceFailed("no hidden state after the prompt")
        }
        var hidden = Array(UnsafeBufferPointer(start: promptHidden, count: nEmbd))

        func drain() throws {
            var rate: Int32 = 0
            var data: UnsafePointer<CChar>?
            var length = 0
            var count: Int64 = 0
            guard mtmd_helper_gen_audio_get_output(gen, &rate, &data, &length, &count) == 0 else {
                throw EngineError.voiceFailed("decoding audio")
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
            onAudio(samples, Int(rate))
        }

        var pending: [UInt8] = []
        var steps = 0
        while true {
            if isCancelled { stats.stopped = true; break }
            if steps >= maxSteps { stats.hitLimit = true; break }
            // The reply ends before the context is full.
            if Int(llama_memory_seq_pos_max(llama_get_memory(ctx), 0)) + 2 >= nCtx { stats.hitLimit = true; break }

            let isText = mtmd_helper_gen_audio_next_is_text(gen)
            let token: llama_token = isText ? llama_sampler_sample(sampler, ctx, -1) : -1
            var next: UnsafePointer<Float>?
            var stop = false
            let rc = hidden.withUnsafeBufferPointer { h in
                mtmd_helper_gen_audio_step_gen(gen, token, h.baseAddress, &next, &stop)
            }
            guard rc == 0 else { throw EngineError.voiceFailed("step \(steps + 1)") }
            steps += 1
            if isText {
                if !stop {
                    stats.textTokens += 1
                    // Control tokens (<|text_end|>...) stay out of the text, as in llama-server.
                    if !llama_vocab_is_control(vocab, token) {
                        pending += piece(token, vocab: vocab)
                        if let text = String(bytes: pending, encoding: .utf8) {
                            if stats.firstTextSeconds == nil { stats.firstTextSeconds = seconds(since: start) }
                            onText(text)
                            pending.removeAll()
                        } else if pending.count > 16 {
                            onText(String(decoding: pending, as: UTF8.self))
                            pending.removeAll()
                        }
                    }
                }
            } else {
                stats.frames += 1
            }
            try drain()
            if stop { break }
            if let next { hidden.withUnsafeMutableBufferPointer { _ = memcpy($0.baseAddress!, next, nEmbd * MemoryLayout<Float>.size) } }
        }
        if !pending.isEmpty { onText(String(decoding: pending, as: UTF8.self)) }
        guard mtmd_helper_gen_audio_flush(gen) == 0 else { throw EngineError.voiceFailed("the last frame") }
        try drain()
        stats.generationSeconds = seconds(since: generationStart)
        return stats
    }

    /// Appends tokens or embedding rows to sequence 0 at explicit positions,
    /// like the realtime slot of llama-server. Logits only for the last one, if asked.
    private func decode(_ ctx: OpaquePointer, model: OpaquePointer, tokens: UnsafePointer<llama_token>?,
                        rows: UnsafePointer<Float>?, count n: Int32, at nPast: inout Int32,
                        logitsLast: Bool = false) throws {
        guard n > 0 else { return }
        if Int(nPast + n) >= nCtx { throw EngineError.promptTooLong(tokens: Int(nPast + n), context: nCtx) }
        let nEmbd = Int(llama_model_n_embd_inp(model))
        let nBatch = Int32(llama_n_batch(ctx))
        var i0: Int32 = 0
        while i0 < n {
            let nCur = min(nBatch, n - i0)
            var batch = llama_batch_init(nCur, rows != nil ? Int32(nEmbd) : 0, 1)
            for i in 0..<Int(nCur) {
                let src = Int(i0) + i
                if let rows {
                    memcpy(batch.embd!.advanced(by: i * nEmbd), rows.advanced(by: src * nEmbd), nEmbd * MemoryLayout<Float>.size)
                } else if let tokens {
                    batch.token![i] = tokens[src]
                }
                batch.pos![i] = nPast + i0 + Int32(i)
                batch.n_seq_id![i] = 1
                batch.seq_id![i]![0] = 0
                batch.logits![i] = logitsLast && src == Int(n) - 1 ? 1 : 0
            }
            batch.n_tokens = nCur
            let rc = llama_decode(ctx, batch)
            llama_batch_free(batch)
            if rc != 0 { throw EngineError.decodeFailed(rc) }
            i0 += nCur
        }
        nPast += n
    }

    /// Decodes the rows the audio stream has ready. Returns how many.
    private func decodeAudioRows(_ stream: OpaquePointer, ctx: OpaquePointer, model: OpaquePointer,
                                 at nPast: inout Int32) throws -> Int {
        var total = 0
        while true {
            let n = mtmd_audio_stream_step(stream, -1)
            if n == -2 { throw EngineError.voiceFailed("encoding the audio") }
            if n <= 0 { return total }
            guard let rows = mtmd_audio_stream_get_output_embd(stream) else {
                throw EngineError.voiceFailed("encoding the audio")
            }
            try decode(ctx, model: model, tokens: nil, rows: rows, count: n, at: &nPast)
            total += Int(n)
        }
    }

    private func bosText() -> String {
        guard let vocab else { return "" }
        return String(cString: llama_vocab_get_text(vocab, llama_vocab_bos(vocab)))
    }

    // MARK: - Helpers

    private func makeSampler(_ s: Sampling, vocab: OpaquePointer, banned: [llama_token],
                             biases extra: [llama_logit_bias] = []) -> UnsafeMutablePointer<llama_sampler> {
        let chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        let biases = banned.map { llama_logit_bias(token: $0, bias: -Float.infinity) } + extra
        if !biases.isEmpty {
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

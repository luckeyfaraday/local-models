import XCTest
@testable import LocalModelsLogic

final class ParsedReplyTests: XCTestCase {
    func testPlainAnswer() {
        let reply = ParsedReply(raw: "Hello.", startsInThink: false)
        XCTAssertNil(reply.thinking)
        XCTAssertEqual(reply.answer, "Hello.")
        XCTAssertTrue(reply.thinkingDone)
    }

    func testThinkingStillOpen() {
        let reply = ParsedReply(raw: "Let me see", startsInThink: true)
        XCTAssertEqual(reply.thinking, "Let me see")
        XCTAssertEqual(reply.answer, "")
        XCTAssertFalse(reply.thinkingDone)
    }

    func testThinkingClosed() {
        let reply = ParsedReply(raw: " 2 + 2 </think>\n\nFour.", startsInThink: true)
        XCTAssertEqual(reply.thinking, "2 + 2")
        XCTAssertEqual(reply.answer, "Four.")
        XCTAssertTrue(reply.thinkingDone)
    }

    func testModelOpensThinkItself() {
        let reply = ParsedReply(raw: "\n<think>hm</think>Yes", startsInThink: false)
        XCTAssertEqual(reply.thinking, "hm")
        XCTAssertEqual(reply.answer, "Yes")
    }

    func testThinkTagInsideAnswerIsText() {
        let reply = ParsedReply(raw: "Use <think> tags", startsInThink: false)
        XCTAssertNil(reply.thinking)
        XCTAssertEqual(reply.answer, "Use <think> tags")
    }
}

final class SpeechChunkTests: XCTestCase {
    private func squeezed(_ s: String) -> String { s.filter { !$0.isWhitespace } }

    func testWordCounts() {
        XCTAssertEqual(SpeechRequest.words("Hello, world"), 2)
        XCTAssertEqual(SpeechRequest.words("  one  two\nthree "), 3)
        XCTAssertEqual(SpeechRequest.words(""), 0)
        // Four characters without spaces are about three spoken words, not one.
        XCTAssertEqual(SpeechRequest.words("你好世界"), 3)
        XCTAssertEqual(SpeechRequest.words("こんにちは"), 4)
    }

    func testShortTextIsOneChunk() {
        XCTAssertEqual(SpeechRequest(text: SpeechRequest.sample).chunks.count, 1)
    }

    func testSentencesAreGrouped() {
        let sentence = "The quick brown fox jumps over the lazy dog near the river bank."
        let text = Array(repeating: sentence, count: 10).joined(separator: " ")
        let chunks = SpeechRequest(text: text).chunks
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks { XCTAssertLessThanOrEqual(SpeechRequest.words(chunk), SpeechRequest.chunkWords) }
        XCTAssertEqual(squeezed(chunks.joined()), squeezed(text))
    }

    func testSentenceWithoutPausesIsCut() {
        let text = (1...100).map { "word\($0)" }.joined(separator: " ") + "."
        let chunks = SpeechRequest(text: text).chunks
        XCTAssertGreaterThanOrEqual(chunks.count, 3)
        for chunk in chunks { XCTAssertLessThanOrEqual(SpeechRequest.words(chunk), SpeechRequest.longSentenceWords) }
        XCTAssertEqual(squeezed(chunks.joined()), squeezed(text))
    }

    func testLongSentenceSplitsAtCommas() {
        let clause = "one two three four five six seven eight nine ten eleven twelve"
        let text = Array(repeating: clause, count: 5).joined(separator: ", ") + "."
        let chunks = SpeechRequest(text: text).chunks
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks.dropLast() { XCTAssertTrue(chunk.hasSuffix(","), chunk) }
    }

    func testChineseIsChunked() {
        let sentence = "今天天气很好我们一起去公园散步然后去喝茶"
        let text = Array(repeating: sentence, count: 12).joined(separator: "，") + "。"
        let chunks = SpeechRequest(text: text).chunks
        XCTAssertGreaterThan(chunks.count, 3)
        for chunk in chunks { XCTAssertLessThanOrEqual(SpeechRequest.words(chunk), SpeechRequest.longSentenceWords) }
        XCTAssertEqual(squeezed(chunks.joined()), squeezed(text))
    }

    func testJapaneseWithoutAnyPunctuationIsCut() {
        let text = String(repeating: "わたしはにほんごをべんきょうしています", count: 10)
        let chunks = SpeechRequest(text: text).chunks
        XCTAssertGreaterThan(chunks.count, 3)
        for chunk in chunks { XCTAssertLessThanOrEqual(SpeechRequest.words(chunk), SpeechRequest.longSentenceWords) }
        XCTAssertEqual(chunks.joined().filter { !$0.isWhitespace }, text)
    }
}

final class VoiceDetectorTests: XCTestCase {
    private func feed(_ detector: inout VoiceDetector, level: Float, seconds: Double, echo: Bool = false) -> [VoiceDetector.Event] {
        let frames = Int((seconds / VoiceDetector.frameSeconds).rounded())
        return (0..<frames).compactMap { _ in detector.process(level: level, echo: echo) }
    }

    func testATurnStartsAndEnds() {
        var detector = VoiceDetector()
        XCTAssertEqual(feed(&detector, level: -80, seconds: 1), [])
        XCTAssertEqual(feed(&detector, level: -20, seconds: 0.5), [.started])
        XCTAssertTrue(detector.inSpeech)
        XCTAssertEqual(feed(&detector, level: -80, seconds: 0.5), [])
        XCTAssertEqual(feed(&detector, level: -80, seconds: 0.4), [.ended])
        XCTAssertFalse(detector.inSpeech)
    }

    func testABlipIsNotATurn() {
        var detector = VoiceDetector()
        _ = feed(&detector, level: -80, seconds: 1)
        XCTAssertEqual(feed(&detector, level: -20, seconds: 0.1), [])
        XCTAssertEqual(feed(&detector, level: -80, seconds: 1), [])
        XCTAssertFalse(detector.inSpeech)
    }

    func testEchoNeedsToBeLouder() {
        var detector = VoiceDetector()
        _ = feed(&detector, level: -80, seconds: 1)
        // Counts as speech in a quiet room, but not over the reply.
        XCTAssertEqual(feed(&detector, level: -45, seconds: 1, echo: true), [])
        XCTAssertEqual(feed(&detector, level: -30, seconds: 0.5, echo: true), [.started])
    }

    func testALongTurnIsCut() {
        var detector = VoiceDetector()
        _ = feed(&detector, level: -80, seconds: 1)
        let events = feed(&detector, level: -20, seconds: VoiceDetector.maxTurnSeconds + 1)
        XCTAssertEqual(events.prefix(2), [.started, .ended])
    }
}

final class WAVTests: XCTestCase {
    func testHeaderAndSamples() {
        let data = WAV.encode([0, 1, -1, 2, -2, 0.5], sampleRate: 24000)
        XCTAssertEqual(data.count, 44 + 12)
        XCTAssertEqual(String(decoding: data[0..<4], as: UTF8.self), "RIFF")
        XCTAssertEqual(String(decoding: data[8..<12], as: UTF8.self), "WAVE")
        func u32(_ at: Int) -> UInt32 { data[at..<(at + 4)].reversed().reduce(0) { $0 << 8 | UInt32($1) } }
        func i16(_ at: Int) -> Int16 { Int16(bitPattern: UInt16(data[at]) | UInt16(data[at + 1]) << 8) }
        XCTAssertEqual(u32(4), 36 + 12)
        XCTAssertEqual(u32(24), 24000)
        XCTAssertEqual(u32(28), 48000)
        XCTAssertEqual(u32(40), 12)
        XCTAssertEqual((0..<6).map { i16(44 + $0 * 2) }, [0, 32767, -32767, 32767, -32767, 16384])
    }
}

final class GGUFNameTests: XCTestCase {
    func testQuantTag() {
        XCTAssertEqual(GGUFName.quantTag("LFM3-Audio-2B-Q4_0.gguf"), "Q4_0")
        XCTAssertEqual(GGUFName.quantTag("LFM3-1.7B-QAD-Q4_0.gguf"), "Q4_0")
        XCTAssertEqual(GGUFName.quantTag("model.q5_k_m.gguf"), "Q5_K_M")
        XCTAssertEqual(GGUFName.quantTag("mmproj-model-f16.gguf"), "F16")
        XCTAssertNil(GGUFName.quantTag("model.gguf"))
    }

    func testProjectorBase() {
        XCTAssertEqual(GGUFName.projectorBase("mmproj-LFM3-Audio-2B-Q8_0.gguf"), "lfm3-audio-2b")
        XCTAssertEqual(GGUFName.projectorBase("LFM3-VL-2B-mmproj-F16.gguf"), "lfm3-vl-2b")
    }
}

final class VoicePromptTests: XCTestCase {
    private let pieces = ["Hello", " there", " my", " friend"]

    func testHeardPartStopsAtAWholeWord() {
        XCTAssertEqual(VoicePrompt.heardPart(of: pieces, seconds: 0.54, secondsPerToken: 0.27), "Hello")
        XCTAssertEqual(VoicePrompt.heardPart(of: pieces, seconds: 0.81, secondsPerToken: 0.27), "Hello there")
    }

    func testHeardPartEnds() {
        XCTAssertEqual(VoicePrompt.heardPart(of: pieces, seconds: 0, secondsPerToken: 0.27), "")
        XCTAssertEqual(VoicePrompt.heardPart(of: pieces, seconds: 60, secondsPerToken: 0.27), "Hello there my friend")
    }

    func testChatPromptKeepsEarlierTurns() {
        let prompt = VoicePrompt.chat(history: [VoiceExchangeText(heard: "Hi", reply: "Hello!")], bos: "<s>")
        XCTAssertTrue(prompt.hasPrefix("<s><|im_start|>system\n"))
        XCTAssertTrue(prompt.contains("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n"))
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>user\n"))
    }
}

final class ChatStoreTests: XCTestCase {
    private var store: ChatStore!

    override func setUp() {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("chats-\(UUID().uuidString)")
        store = ChatStore(directory: folder)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: store.directory)
    }

    private func chat(_ text: String, at seconds: Double, images: [String] = []) -> StoredChat {
        let stats = GenerationStats(promptTokens: 12, promptSeconds: 0.5, generatedTokens: 30, generationSeconds: 1.5)
        return StoredChat(id: UUID(), updated: Date(timeIntervalSinceReferenceDate: seconds), model: "LFM3", messages: [
            StoredMessage(id: UUID(), role: .user, raw: text, images: images),
            StoredMessage(id: UUID(), role: .assistant, raw: "<think>hm</think>Fine.", startsInThink: false, stats: stats),
        ])
    }

    func testRoundTripWithImage() throws {
        let saved = chat("What is this?", at: 1000, images: ["a.jpg"])
        let jpeg = Data([0xFF, 0xD8, 0xFF, 0xD9])
        try store.save(saved, images: ["a.jpg": jpeg])
        XCTAssertEqual(try store.load(saved.id), saved)
        XCTAssertEqual(store.image("a.jpg", in: saved.id), jpeg)
        XCTAssertNil(store.image("b.jpg", in: saved.id))
    }

    func testSavingAgainReplaces() throws {
        var saved = chat("First", at: 1000)
        try store.save(saved)
        saved.messages.append(StoredMessage(id: UUID(), role: .user, raw: "More"))
        saved.updated = Date(timeIntervalSinceReferenceDate: 2000)
        try store.save(saved)
        XCTAssertEqual(try store.load(saved.id).messages.count, 3)
        XCTAssertEqual(store.summaries().map(\.messages), [3])
    }

    func testSummariesNewestFirstAndDelete() throws {
        let old = chat("Old", at: 1000), new = chat("New", at: 2000)
        try store.save(old)
        try store.save(new)
        XCTAssertEqual(store.summaries().map(\.title), ["New", "Old"])
        store.delete(new.id)
        XCTAssertEqual(store.summaries().map(\.id), [old.id])
        XCTAssertThrowsError(try store.load(new.id))
    }

    func testEmptyStore() {
        XCTAssertEqual(store.summaries(), [])
    }

    func testTitles() {
        XCTAssertEqual(chat("  Hello there \nsecond line", at: 0).title, "Hello there")
        XCTAssertEqual(chat("", at: 0, images: ["a.jpg"]).title, "Image")
        let long = chat(String(repeating: "word ", count: 40), at: 0).title
        XCTAssertLessThanOrEqual(long.count, 61)
        XCTAssertTrue(long.hasSuffix("…"))
    }
}

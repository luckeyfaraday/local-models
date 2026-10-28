import Foundation
import NaturalLanguage

struct SpeechRequest: Equatable {
    var text: String
    var voice: String = "female"
    var language: String = "en"

    /// LFM3-Audio makes 80 ms of audio per frame: 1,500 frames is two minutes.
    static let maxFrames = 1500

    /// Spoken in one go, LFM3-Audio starts to slur, skip and repeat itself after
    /// about 40 words (~12 s). So text is spoken a few sentences at a time, each
    /// with a fresh prompt.
    static let chunkWords = 30
    static let longSentenceWords = 40

    var chunks: [String] {
        var sentences: [String] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { sentences.append(sentence) }
            return true
        }
        // A very long sentence is split where it pauses: after a comma, semicolon or colon.
        let pieces = sentences.flatMap { sentence -> [String] in
            guard Self.words(sentence) > Self.longSentenceWords else { return [sentence] }
            return sentence.replacingOccurrences(of: #"([,;:])\s+"#, with: "$1\n", options: .regularExpression)
                .split(separator: "\n").map(String.init)
        }
        var chunks: [String] = []
        var current = ""
        for piece in pieces {
            if !current.isEmpty && Self.words(current) + Self.words(piece) > Self.chunkWords {
                chunks.append(current)
                current = piece
            } else {
                current = current.isEmpty ? piece : current + " " + piece
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks.isEmpty ? [text] : chunks
    }

    private static func words(_ s: String) -> Int {
        s.split(whereSeparator: \.isWhitespace).count
    }

    static let sample = "Hi! I'm a language model running entirely on this iPhone. Nothing you say to me leaves the phone, and I don't need an internet connection to talk back."

    /// The "tts" chat template of LFM3-Audio, which llama.cpp's C API can't render.
    static func lfm3Prompt(_ request: SpeechRequest, bos: String) -> String {
        let language = SpeechLanguage.name(for: request.language)
        let voice = request.voice.lowercased() == "male" ? "male" : "female"
        return bos
            + "<|im_start|>system\nPerform TTS in \(language) using a \(voice) voice.<|im_end|>\n"
            + "<|im_start|>user\n\(request.text)<|im_end|>\n"
            + "<|im_start|>assistant\n<|audio_start|>"
    }
}

struct SpeechLanguage: Identifiable, Hashable {
    let code: String
    let name: String
    var id: String { code }

    /// The languages LFM3-Audio's template knows, in its own words.
    static let all: [SpeechLanguage] = [
        ("en", "American English"), ("en-gb", "British English"), ("es", "Spanish"), ("fr", "French"),
        ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("sv", "Swedish"),
        ("da", "Danish"), ("fi", "Finnish"), ("pl", "Polish"), ("el", "Greek"), ("tr", "Turkish"),
        ("ar", "Arabic"), ("he", "Hebrew"), ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
    ].map { SpeechLanguage(code: $0.0, name: $0.1) }

    static func name(for code: String) -> String {
        all.first { $0.code == code.lowercased() }?.name ?? code
    }
}

struct SpeechStats: Equatable {
    var promptTokens = 0
    var promptSeconds = 0.0
    var frames = 0
    var generationSeconds = 0.0
    /// From the start of the request to the first audio sample.
    var firstAudioSeconds: Double?
    var samples = 0
    var sampleRate = 0
    var stopped = false
    var hitFrameLimit = false

    /// Pieces the text was spoken in.
    var chunks = 1

    var audioSeconds: Double { sampleRate > 0 ? Double(samples) / Double(sampleRate) : 0 }
    var totalSeconds: Double { promptSeconds + generationSeconds }
    /// Seconds of audio made per second of work: above 1 keeps up with playback.
    var realTimeFactor: Double { totalSeconds > 0 ? audioSeconds / totalSeconds : 0 }
    var framesPerSecond: Double { generationSeconds > 0 ? Double(frames) / generationSeconds : 0 }

    /// Adds the next piece of a request spoken in several pieces.
    mutating func append(_ next: SpeechStats) {
        promptTokens += next.promptTokens
        promptSeconds += next.promptSeconds
        frames += next.frames
        generationSeconds += next.generationSeconds
        samples += next.samples
        if next.sampleRate > 0 { sampleRate = next.sampleRate }
        stopped = stopped || next.stopped
        hitFrameLimit = hitFrameLimit || next.hitFrameLimit
    }
}

enum SpeechEvent {
    case audio([Float], sampleRate: Int)
    case done(SpeechStats)
}

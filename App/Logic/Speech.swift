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
        let pieces = sentences.flatMap(Self.split)
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

    /// A very long sentence is split where it pauses: after a comma, semicolon
    /// or colon. A stretch with no pause in it is cut every `chunkWords` words.
    static func split(_ sentence: String) -> [String] {
        guard words(sentence) > longSentenceWords else { return [sentence] }
        return sentence
            .replacingOccurrences(of: #"([,;:])\s+|([、，；：])\s*"#, with: "$1$2\n", options: .regularExpression)
            .split(separator: "\n")
            .flatMap { part in words(String(part)) > longSentenceWords ? cut(String(part)) : [String(part)] }
    }

    /// The length of a text in spoken words. Chinese and Japanese are written
    /// without spaces, so there each character counts: about 0.7 of a word, by
    /// how long it takes to say.
    static func words(_ s: String) -> Int {
        var tenths = 0
        var inWord = false
        for character in s {
            if character.isWhitespace {
                inWord = false
            } else if isUnspaced(character) {
                tenths += 7
                inWord = false
            } else if !inWord && !character.isPunctuation && !character.isSymbol {
                tenths += 10
                inWord = true
            }
        }
        return (tenths + 9) / 10
    }

    /// Cuts a text with no pauses into pieces of about `chunkWords` words:
    /// between words, or between characters where there are no spaces.
    private static func cut(_ text: String) -> [String] {
        var parts: [String] = []
        var current = ""
        func close() {
            let part = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !part.isEmpty { parts.append(part) }
            current = ""
        }
        for character in text {
            if (character.isWhitespace || isUnspaced(character)) && words(current) >= chunkWords { close() }
            current.append(character)
        }
        close()
        return parts
    }

    /// Han, hiragana and katakana: scripts written without spaces between words.
    private static func isUnspaced(_ character: Character) -> Bool {
        guard let scalar = character.unicodeScalars.first?.value else { return false }
        switch scalar {
        case 0x3040...0x30FF, 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF, 0x20000...0x2FA1F: return true
        default: return false
        }
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

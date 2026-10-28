import Foundation

struct SpeechRequest: Equatable {
    var text: String
    var voice: String = "female"
    var language: String = "en"

    /// LFM3-Audio makes 80 ms of audio per frame: 1,500 frames is two minutes.
    static let maxFrames = 1500

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

    var audioSeconds: Double { sampleRate > 0 ? Double(samples) / Double(sampleRate) : 0 }
    var totalSeconds: Double { promptSeconds + generationSeconds }
    /// Seconds of audio made per second of work: above 1 keeps up with playback.
    var realTimeFactor: Double { totalSeconds > 0 ? audioSeconds / totalSeconds : 0 }
    var framesPerSecond: Double { generationSeconds > 0 ? Double(frames) / generationSeconds : 0 }
}

enum SpeechEvent {
    case audio([Float], sampleRate: Int)
    case done(SpeechStats)
}

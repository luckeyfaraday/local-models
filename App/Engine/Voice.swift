import Foundation

/// One earlier exchange of a voice conversation, as text: what the user said
/// (transcribed after the reply) and what the model answered.
struct VoiceExchangeText: Equatable {
    var heard: String
    var reply: String
}

/// The prompts of a spoken conversation with LFM3-Audio, as llama-server's
/// /v1/realtime renders them from the model's chat template: a system turn,
/// the earlier turns, then the user's audio and an assistant turn that opens
/// with `<|mixed_start|>` (added by the gen-audio pipeline in "mixed" mode).
enum VoicePrompt {
    /// The system prompt of LFM3-Audio's model card for interleaved text and speech.
    static let system = "Respond with interleaved text and audio."

    /// Closes the user's audio and opens the reply.
    static let replyOpen = "<|im_end|>\n<|im_start|>assistant\n"

    /// A reply is cut after this many steps (text tokens + 80 ms audio frames): about two minutes.
    static let maxSteps = 1800

    /// Everything before the user's audio.
    static func chat(history: [VoiceExchangeText], bos: String) -> String {
        var s = bos + "<|im_start|>system\n\(system)<|im_end|>\n"
        for turn in history {
            s += "<|im_start|>user\n\(turn.heard)<|im_end|>\n"
            s += "<|im_start|>assistant\n\(turn.reply)<|im_end|>\n"
        }
        return s + "<|im_start|>user\n"
    }

    /// Transcription of one clip, llama.cpp's ASR preset for LFM3 (`Perform ASR in English.`).
    static func asr(language: String, bos: String) -> String {
        let name = language.lowercased().hasPrefix("en") ? "English" : SpeechLanguage.name(for: language)
        return bos + "<|im_start|>system\nPerform ASR in \(name).<|im_end|>\n<|im_start|>user\n"
    }
}

struct VoiceStats: Equatable {
    /// Positions read before the reply: text tokens and audio rows.
    var promptTokens = 0
    var audioRows = 0
    /// Reading the conversation and encoding the user's audio.
    var promptSeconds = 0.0
    var textTokens = 0
    var frames = 0
    var generationSeconds = 0.0
    /// From the start of the request.
    var firstTextSeconds: Double?
    var firstAudioSeconds: Double?
    var samples = 0
    var sampleRate = 0
    var stopped = false
    var hitLimit = false

    var audioSeconds: Double { sampleRate > 0 ? Double(samples) / Double(sampleRate) : 0 }
    /// Seconds of speech made per second of generation: above 1 keeps up with playback.
    var realTimeFactor: Double { generationSeconds > 0 ? audioSeconds / generationSeconds : 0 }
}

enum VoiceEvent {
    case text(String)
    case audio([Float], sampleRate: Int)
    case done(VoiceStats)
}

import Foundation

enum SamplingPreset: String, CaseIterable, Identifiable {
    case general, math, greedy

    var id: String { rawValue }

    var label: String {
        switch self {
        case .general: return "Chat, tools & agents"
        case .math: return "Math & science"
        case .greedy: return "Deterministic"
        }
    }

    var detail: String {
        switch self {
        case .general: return "temp 0.1 · top-k 50 · repeat 1.1 (Liquid's default)"
        case .math: return "temp 1.0 · top-p 0.95 · top-k 20"
        case .greedy: return "always the most likely token"
        }
    }

    var sampling: Sampling {
        switch self {
        case .general: return Sampling(temperature: 0.1, topK: 50, topP: 1.0, repeatPenalty: 1.1)
        case .math: return Sampling(temperature: 1.0, topK: 20, topP: 0.95, repeatPenalty: 1.0)
        case .greedy: return Sampling(temperature: 0, topK: 0, topP: 1.0, repeatPenalty: 1.0)
        }
    }
}

/// UserDefaults keys shared by the settings screen (@AppStorage) and the chat.
enum SettingKey {
    static let thinking = "thinking"
    static let contextLength = "contextLength"
    static let preset = "samplingPreset"
    static let maxTokens = "maxTokens"
    static let systemPrompt = "systemPrompt"
    static let lastModel = "lastModel"
    static let speechVoice = "speechVoice"
    static let speechLanguage = "speechLanguage"
    static let speechText = "speechText"
}

struct AppSettings {
    var thinking: Bool
    var contextLength: Int
    var preset: SamplingPreset
    var maxTokens: Int
    var systemPrompt: String

    static let contextChoices = [4096, 8192, 16384, 32768]

    static func register() {
        UserDefaults.standard.register(defaults: [
            SettingKey.thinking: false,
            SettingKey.contextLength: 8192,
            SettingKey.preset: SamplingPreset.general.rawValue,
            SettingKey.maxTokens: 4096,
            SettingKey.systemPrompt: "",
        ])
    }

    static var current: AppSettings {
        let d = UserDefaults.standard
        return AppSettings(
            thinking: d.bool(forKey: SettingKey.thinking),
            contextLength: d.integer(forKey: SettingKey.contextLength),
            preset: SamplingPreset(rawValue: d.string(forKey: SettingKey.preset) ?? "") ?? .general,
            maxTokens: d.integer(forKey: SettingKey.maxTokens),
            systemPrompt: d.string(forKey: SettingKey.systemPrompt) ?? ""
        )
    }
}

import Foundation

enum Role: String, Codable {
    case system, user, assistant
}

struct GenerationStats: Equatable, Codable {
    var promptTokens = 0
    var promptSeconds = 0.0
    var generatedTokens = 0
    var generationSeconds = 0.0
    var hitContextLimit = false
    /// Images read with the prompt. Their encoding counts in `promptSeconds`.
    var images = 0

    var tokensPerSecond: Double { generationSeconds > 0 ? Double(generatedTokens) / generationSeconds : 0 }
    var promptTokensPerSecond: Double { promptSeconds > 0 ? Double(promptTokens) / promptSeconds : 0 }
}

/// Splits raw model output into reasoning and answer.
struct ParsedReply: Equatable {
    var thinking: String?
    var answer: String
    var thinkingDone: Bool

    init(raw: String, startsInThink: Bool) {
        var rest = Substring(raw)
        var inThink = startsInThink
        if !inThink, let open = rest.range(of: "<think>"),
           rest[..<open.lowerBound].allSatisfy(\.isWhitespace) {
            inThink = true
            rest = rest[open.upperBound...]
        }
        guard inThink else {
            thinking = nil
            answer = String(raw)
            thinkingDone = true
            return
        }
        if let close = rest.range(of: "</think>") {
            thinking = rest[..<close.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            answer = String(rest[close.upperBound...].drop { $0.isWhitespace })
            thinkingDone = true
        } else {
            thinking = rest.trimmingCharacters(in: .whitespacesAndNewlines)
            answer = ""
            thinkingDone = false
        }
    }
}

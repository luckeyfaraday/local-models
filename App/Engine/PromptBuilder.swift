import Foundation
import llama

enum Role: String {
    case system, user, assistant
}

struct Turn {
    let role: Role
    let content: String
}

struct RenderedPrompt {
    let text: String
    /// The prompt already opened a `<think>` block, so output starts as reasoning.
    let startsInThink: Bool
}

/// Turns a chat into the exact prompt text a model expects.
///
/// llama.cpp's C API only knows a fixed list of built-in templates and has no
/// switch for reasoning, so ChatML models (LFM, Qwen, ...) are formatted here.
/// Everything else goes through `llama_chat_apply_template`.
struct PromptBuilder {
    enum Style: Equatable {
        /// LFM3: `<think>` opens the reply, `<think></think>` turns reasoning off.
        case lfm3
        /// ChatML with Qwen-style reasoning: the model opens `<think>` itself.
        case chatmlThinking
        case chatml
        case builtin(String)
    }

    let style: Style
    let addBOS: Bool

    init(template: String?, vocabAddsBOS: Bool) {
        let t = template ?? ""
        if t.contains("<|im_start|>") {
            if t.contains("<think></think>") && t.contains("enable_thinking") {
                style = .lfm3
            } else if t.contains("<think>") {
                style = .chatmlThinking
            } else {
                style = .chatml
            }
        } else if t.isEmpty {
            style = .chatml
        } else {
            style = .builtin(t)
        }
        addBOS = vocabAddsBOS || t.contains("bos_token")
    }

    var supportsThinking: Bool {
        style == .lfm3 || style == .chatmlThinking
    }

    func render(_ turns: [Turn], system: String, thinking: Bool) -> RenderedPrompt {
        let think = thinking && supportsThinking
        let system = system.trimmingCharacters(in: .whitespacesAndNewlines)

        if case .builtin(let template) = style {
            var all = turns
            if !system.isEmpty { all.insert(Turn(role: .system, content: system), at: 0) }
            if let text = Self.applyBuiltin(template: template, turns: all) {
                return RenderedPrompt(text: text, startsInThink: false)
            }
        }

        var s = ""
        if !system.isEmpty { s += "<|im_start|>system\n\(system)<|im_end|>\n" }
        for turn in turns {
            switch turn.role {
            case .system:
                continue
            case .user:
                s += "<|im_start|>user\n\(turn.content)<|im_end|>\n"
            case .assistant:
                // Earlier reasoning is dropped, as the model's own template does.
                let prefix = style == .lfm3 ? "<think></think>" : ""
                s += "<|im_start|>assistant\n\(prefix)\(turn.content)<|im_end|>\n"
            }
        }
        s += "<|im_start|>assistant\n"
        switch style {
        case .lfm3:
            s += think ? "<think>" : "<think></think>"
            return RenderedPrompt(text: s, startsInThink: think)
        case .chatmlThinking:
            if !think { s += "<think>\n\n</think>\n\n" }
            return RenderedPrompt(text: s, startsInThink: false)
        default:
            return RenderedPrompt(text: s, startsInThink: false)
        }
    }

    private static func applyBuiltin(template: String, turns: [Turn]) -> String? {
        var owned: [UnsafeMutablePointer<CChar>] = []
        defer { owned.forEach { free($0) } }
        var messages: [llama_chat_message] = []
        for turn in turns {
            guard let role = strdup(turn.role.rawValue), let content = strdup(turn.content) else { return nil }
            owned += [role, content]
            messages.append(llama_chat_message(role: UnsafePointer(role), content: UnsafePointer(content)))
        }
        let count = messages.count
        var capacity = 4096
        var buffer = [CChar](repeating: 0, count: capacity)
        var n = llama_chat_apply_template(template, &messages, count, true, &buffer, Int32(capacity))
        if n < 0 { return nil }
        if Int(n) > capacity {
            capacity = Int(n)
            buffer = [CChar](repeating: 0, count: capacity)
            n = llama_chat_apply_template(template, &messages, count, true, &buffer, Int32(capacity))
            if n < 0 { return nil }
        }
        return String(decoding: buffer.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
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

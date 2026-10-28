import SwiftUI

struct MessageRow: View {
    let message: ChatViewModel.Message
    let isStreaming: Bool
    @State private var showThinking = false

    var body: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 48)
                Text(message.raw)
                    .textSelection(.enabled)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .foregroundStyle(.white)
                    .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        } else {
            assistant
        }
    }

    private var assistant: some View {
        let reply = message.reply
        return VStack(alignment: .leading, spacing: 8) {
            if let thinking = reply.thinking {
                DisclosureGroup(isExpanded: $showThinking) {
                    Text(thinking.isEmpty ? "…" : thinking)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                } label: {
                    HStack(spacing: 6) {
                        if !reply.thinkingDone && isStreaming { ProgressView().controlSize(.mini) }
                        Text(reply.thinkingDone ? "Thoughts" : "Thinking…")
                    }
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)
                }
                .tint(.secondary)
            }

            if !reply.answer.isEmpty {
                Text(markdown(reply.answer))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if isStreaming && reply.thinking == nil {
                ProgressView().controlSize(.small)
            }

            if let error = message.error {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }

            if let footer {
                Text(footer)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var footer: String? {
        guard let stats = message.stats else { return message.stopped ? "Stopped" : nil }
        var parts = [String(format: "%d tok · %.1f tok/s", stats.generatedTokens, stats.tokensPerSecond)]
        parts.append(String(format: "prompt %d tok in %.1fs", stats.promptTokens, stats.promptSeconds))
        if message.stopped { parts.append("stopped") }
        if stats.hitContextLimit { parts.append("context full") }
        return parts.joined(separator: " · ")
    }

    private func markdown(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }
}

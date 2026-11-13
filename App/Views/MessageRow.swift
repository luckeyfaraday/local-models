import SwiftUI

struct MessageRow: View {
    let message: ChatViewModel.Message
    let isStreaming: Bool
    @State private var showThinking = false

    var body: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 48)
                VStack(alignment: .trailing, spacing: 6) {
                    ForEach(message.images) { image in
                        Image(uiImage: image.thumbnail)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: 220, maxHeight: 280)
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .accessibilityLabel("Attached image")
                    }
                    if !message.raw.isEmpty {
                        Text(message.raw)
                            .textSelection(.enabled)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .foregroundStyle(.white)
                            .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    }
                }
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
                        Text(thinkingLabel(thinking, done: reply.thinkingDone))
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
        let images = stats.images == 0 ? "" : stats.images == 1 ? " with image" : " with \(stats.images) images"
        let kept = (stats.reusedTokens ?? 0) > 0 ? " (\(stats.reusedTokens ?? 0) kept)" : ""
        parts.append(String(format: "prompt %d tok%@%@ in %.1fs", stats.promptTokens, kept, images, stats.promptSeconds))
        if message.stopped { parts.append("stopped") }
        if stats.hitContextLimit { parts.append("context full") }
        return parts.joined(separator: " · ")
    }

    private func thinkingLabel(_ thinking: String, done: Bool) -> String {
        let words = thinking.split(whereSeparator: \.isWhitespace).count
        let count = words == 0 ? "" : " · \(words) words"
        return (done ? "Thoughts" : "Thinking…") + count
    }

    private func markdown(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }
}

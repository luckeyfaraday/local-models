import SwiftUI

/// The saved chats, newest first. Tap one to open it, swipe to delete.
struct HistoryView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                ForEach(chat.history) { summary in
                    Button {
                        Task {
                            await chat.open(summary.id)
                            dismiss()
                        }
                    } label: {
                        row(summary)
                    }
                    .foregroundStyle(.primary)
                }
                .onDelete { offsets in
                    offsets.map { chat.history[$0].id }.forEach { chat.delete($0) }
                }
            }
            .overlay {
                if chat.history.isEmpty {
                    ContentUnavailableView("No chats yet", systemImage: "bubble.left.and.bubble.right",
                                           description: Text("Chats are saved on this iPhone as you write."))
                }
            }
            .navigationTitle("Chats")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func row(_ summary: ChatSummary) -> some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text(summary.title).lineLimit(2)
                Text(detail(summary)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 0)
            if summary.id == chat.chatID {
                Image(systemName: "checkmark").font(.footnote.weight(.semibold)).foregroundStyle(.tint)
            }
        }
        .contentShape(Rectangle())
    }

    private func detail(_ summary: ChatSummary) -> String {
        let when = summary.updated.formatted(.relative(presentation: .named))
        return [when, summary.model].compactMap { $0 }.joined(separator: " · ")
    }
}

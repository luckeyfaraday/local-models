import Foundation

/// One message of a saved chat. Images are file names next to the chat's JSON.
struct StoredMessage: Codable, Equatable {
    var id: UUID
    var role: Role
    var raw: String
    var images: [String] = []
    var startsInThink = false
    var stats: GenerationStats?
    var error: String?
    var stopped = false
}

struct StoredChat: Codable, Equatable {
    var id: UUID
    var updated: Date
    /// The model that answered last, for the list.
    var model: String?
    var messages: [StoredMessage]

    /// The first thing the user wrote, cut to a line.
    var title: String {
        let first = messages.first { $0.role == .user }
        let text = (first?.raw ?? "").split(whereSeparator: \.isNewline).first.map(String.init) ?? ""
        let line = text.trimmingCharacters(in: .whitespaces)
        if line.isEmpty { return first?.images.isEmpty == false ? "Image" : "Chat" }
        return line.count > 60 ? String(line.prefix(60)).trimmingCharacters(in: .whitespaces) + "…" : line
    }

    var summary: ChatSummary {
        ChatSummary(id: id, title: title, updated: updated, model: model, messages: messages.count)
    }
}

/// A row of the chat list.
struct ChatSummary: Codable, Equatable, Identifiable {
    var id: UUID
    var title: String
    var updated: Date
    var model: String?
    var messages: Int
}

/// Chats on disk: one folder each, holding `chat.json`, `summary.json` (what
/// the list shows, so it doesn't read every chat) and the attached images as
/// JPEG files. iOS readily kills an app that holds a model in memory, so a
/// chat is written whenever it changes.
struct ChatStore {
    let directory: URL

    /// Application Support/Chats: kept in backups, not shown in Files.
    static var standard: ChatStore {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return ChatStore(directory: base.appendingPathComponent("Chats", isDirectory: true))
    }

    private func folder(_ id: UUID) -> URL {
        directory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    /// Writes the chat, and the images that aren't on disk yet (by file name).
    func save(_ chat: StoredChat, images: [String: Data] = [:]) throws {
        let folder = folder(chat.id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for (name, data) in images {
            try data.write(to: folder.appendingPathComponent(name), options: .atomic)
        }
        try JSONEncoder().encode(chat).write(to: folder.appendingPathComponent("chat.json"), options: .atomic)
        try JSONEncoder().encode(chat.summary).write(to: folder.appendingPathComponent("summary.json"), options: .atomic)
    }

    func load(_ id: UUID) throws -> StoredChat {
        try JSONDecoder().decode(StoredChat.self, from: Data(contentsOf: folder(id).appendingPathComponent("chat.json")))
    }

    func image(_ name: String, in id: UUID) -> Data? {
        try? Data(contentsOf: folder(id).appendingPathComponent(name))
    }

    /// Every saved chat, newest first.
    func summaries() -> [ChatSummary] {
        let folders = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        let all = folders.compactMap { folder -> ChatSummary? in
            if let data = try? Data(contentsOf: folder.appendingPathComponent("summary.json")),
               let summary = try? JSONDecoder().decode(ChatSummary.self, from: data) {
                return summary
            }
            guard let data = try? Data(contentsOf: folder.appendingPathComponent("chat.json")) else { return nil }
            return (try? JSONDecoder().decode(StoredChat.self, from: data))?.summary
        }
        return all.sorted { $0.updated > $1.updated }
    }

    func delete(_ id: UUID) {
        try? FileManager.default.removeItem(at: folder(id))
    }
}

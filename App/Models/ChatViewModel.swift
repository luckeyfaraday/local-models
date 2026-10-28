import Foundation
import Observation
import UIKit

@Observable @MainActor
final class ChatViewModel {
    enum Status: Equatable {
        case noModel
        case loading(String)
        case ready
        case generating
        case failed(String)
    }

    struct Message: Identifiable, Equatable {
        let id = UUID()
        let role: Role
        var raw: String
        var startsInThink = false
        var stats: GenerationStats?
        var error: String?
        var stopped = false

        var reply: ParsedReply { ParsedReply(raw: raw, startsInThink: startsInThink) }
    }

    private(set) var messages: [Message] = []
    private(set) var status: Status = .noModel
    private(set) var info: ModelInfo?
    private(set) var loadedFile: String?

    @ObservationIgnored private let engine = LlamaEngine()
    @ObservationIgnored private var builder: PromptBuilder?
    @ObservationIgnored private var loadedContextSetting = 0
    @ObservationIgnored private var generation: Task<Void, Never>?
    @ObservationIgnored private var stopRequested = false
    @ObservationIgnored private var didAutoload = false

    var isGenerating: Bool { status == .generating }
    var isBusy: Bool {
        if case .loading = status { return true }
        return status == .generating
    }
    var supportsThinking: Bool { builder?.supportsThinking ?? false }

    // MARK: - Models

    func autoload(from store: ModelStore) async {
        guard !didAutoload else { return }
        didAutoload = true
        let last = UserDefaults.standard.string(forKey: SettingKey.lastModel)
        if let model = last.flatMap(store.model(named:)) ?? (store.models.count == 1 ? store.models.first : nil) {
            await load(model)
        }
    }

    func load(_ model: ModelStore.LocalModel) async {
        stop()
        status = .loading(model.displayName)
        let contextLength = AppSettings.current.contextLength
        do {
            let info = try await engine.load(path: model.url.path, contextLength: contextLength)
            self.info = info
            builder = PromptBuilder(template: info.chatTemplate, vocabAddsBOS: info.addsBOS)
            loadedFile = model.fileName
            loadedContextSetting = contextLength
            UserDefaults.standard.set(model.fileName, forKey: SettingKey.lastModel)
            status = .ready
        } catch {
            info = nil
            builder = nil
            loadedFile = nil
            status = .failed(error.localizedDescription)
        }
    }

    func unloadIfDeleted(_ model: ModelStore.LocalModel) async {
        guard loadedFile == model.fileName else { return }
        stop()
        await engine.unload()
        info = nil
        builder = nil
        loadedFile = nil
        status = .noModel
    }

    // MARK: - Chat

    func send(_ text: String, store: ModelStore) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isBusy, loadedFile != nil else { return }
        messages.append(Message(role: .user, raw: text))

        generation = Task {
            let settings = AppSettings.current
            // Context length is fixed when the context is created: reload if it changed.
            if settings.contextLength != loadedContextSetting, let file = loadedFile, let model = store.model(named: file) {
                await load(model)
            }
            guard let builder else { return }

            let turns = messages.compactMap { m -> Turn? in
                if m.role == .assistant {
                    let answer = m.reply.answer
                    return answer.isEmpty ? nil : Turn(role: .assistant, content: answer)
                }
                return Turn(role: m.role, content: m.raw)
            }
            let prompt = builder.render(turns, system: settings.systemPrompt, thinking: settings.thinking)
            let reply = Message(role: .assistant, raw: "", startsInThink: prompt.startsInThink)
            let id = reply.id
            messages.append(reply)
            status = .generating
            stopRequested = false
            UIApplication.shared.isIdleTimerDisabled = true
            defer {
                UIApplication.shared.isIdleTimerDisabled = false
                if status == .generating { status = .ready }
                generation = nil
            }

            // Redrawing a long reply on every token costs more than generating it,
            // so text reaches the screen at most every 50 ms.
            var pending = ""
            var lastFlush = Date.distantPast

            do {
                let stream = engine.generate(prompt: prompt.text, addBOS: builder.addBOS,
                                             sampling: settings.preset.sampling, maxTokens: settings.maxTokens,
                                             banned: prompt.banned)
                for try await event in stream {
                    guard messages.contains(where: { $0.id == id }) else { break }
                    switch event {
                    case .text(let piece):
                        pending += piece
                        if Date().timeIntervalSince(lastFlush) >= 0.05 {
                            append(pending, to: id)
                            pending = ""
                            lastFlush = Date()
                        }
                    case .done(let stats):
                        append(pending, to: id)
                        pending = ""
                        if let i = messages.firstIndex(where: { $0.id == id }) { messages[i].stats = stats }
                    }
                }
                append(pending, to: id)
                if stopRequested, let i = messages.firstIndex(where: { $0.id == id }) {
                    messages[i].stopped = true
                }
            } catch {
                append(pending, to: id)
                if let i = messages.firstIndex(where: { $0.id == id }) {
                    messages[i].error = error.localizedDescription
                }
            }
        }
    }

    private func append(_ text: String, to id: UUID) {
        guard !text.isEmpty, let i = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[i].raw += text
    }

    /// Ends the reply at the next token. The stream still finishes normally,
    /// so the stopped message keeps its tokens/sec.
    func stop() {
        guard generation != nil else { return }
        stopRequested = true
        engine.cancel()
    }

    func newChat() {
        stop()
        messages.removeAll()
    }
}

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
        var images: [ChatImage] = []
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

    @ObservationIgnored private let engine: LlamaEngine
    /// Text to speech, for models that can talk. Shares the loaded model.
    @ObservationIgnored let speech: SpeechViewModel
    /// Spoken conversation, for models that hear and speak (LFM3-Audio).
    @ObservationIgnored let voice: VoiceViewModel
    @ObservationIgnored private var builder: PromptBuilder?
    @ObservationIgnored private var loadedContextSetting = 0
    @ObservationIgnored private var generation: Task<Void, Never>?
    @ObservationIgnored private var stopRequested = false
    @ObservationIgnored private var didAutoload = false
    @ObservationIgnored private var backgroundTask = UIBackgroundTaskIdentifier.invalid

    init() {
        engine = LlamaEngine()
        speech = SpeechViewModel(engine: engine)
        voice = VoiceViewModel(engine: engine)
    }

    var isGenerating: Bool { status == .generating }
    var isBusy: Bool {
        if case .loading = status { return true }
        return status == .generating || speech.isSpeaking || voice.isBusy
    }
    var supportsThinking: Bool { builder?.supportsThinking ?? false }
    var seesImages: Bool { info?.seesImages == true }

    // MARK: - Models

    func autoload(from store: ModelStore) async {
        guard !didAutoload else { return }
        didAutoload = true
        let defaults = UserDefaults.standard
        // A load that never finished most likely got the app killed for memory.
        // Loading the same model again on every launch would make it unusable.
        if let unfinished = defaults.string(forKey: SettingKey.loadingModel) {
            defaults.removeObject(forKey: SettingKey.loadingModel)
            let name = store.model(named: unfinished)?.displayName ?? unfinished
            status = .failed("\(name) didn't finish loading last time, so it wasn't loaded again. "
                + "Pick it to retry, or choose a smaller context length in Settings first.")
            return
        }
        let last = defaults.string(forKey: SettingKey.lastModel)
        if let model = last.flatMap(store.model(named:)) ?? (store.models.count == 1 ? store.models.first : nil) {
            await load(model)
        }
    }

    func load(_ model: ModelStore.LocalModel) async {
        stop()
        speech.stop()
        voice.close()
        status = .loading(model.displayName)
        let contextLength = AppSettings.current.contextLength
        let projector = ModelStore.shared.projector(for: model)
        // Cleared when the load returns, either way. Still set at the next launch: see autoload.
        UserDefaults.standard.set(model.fileName, forKey: SettingKey.loadingModel)
        defer { UserDefaults.standard.removeObject(forKey: SettingKey.loadingModel) }
        do {
            let info = try await engine.load(path: model.url.path, projector: projector?.url.path,
                                             contextLength: contextLength)
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
        speech.stop()
        voice.close()
        await engine.unload()
        info = nil
        builder = nil
        loadedFile = nil
        status = .noModel
    }

    // MARK: - Chat

    func send(_ text: String, images: [ChatImage] = [], store: ModelStore) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || !images.isEmpty, !isBusy, loadedFile != nil else { return }
        messages.append(Message(role: .user, raw: text, images: images))

        generation = Task {
            defer { generation = nil }
            let settings = AppSettings.current
            // Context length is fixed when the context is created: reload if it changed.
            if settings.contextLength != loadedContextSetting, let file = loadedFile, let model = store.model(named: file) {
                await load(model)
            }
            guard let builder else { return }

            // Every turn re-reads the whole chat, earlier images included. A model
            // without vision gets the text alone.
            let sees = seesImages
            var images: [Data] = []
            let turns = messages.compactMap { m -> Turn? in
                if m.role == .assistant {
                    let answer = m.reply.answer
                    return answer.isEmpty ? nil : Turn(role: .assistant, content: answer)
                }
                let attached = sees ? m.images : []
                images += attached.map(\.jpeg)
                return Turn(role: m.role, content: m.raw, images: attached.count)
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
            }

            // Redrawing a long reply on every token costs more than generating it,
            // so text reaches the screen at most every 50 ms.
            var pending = ""
            var lastFlush = Date.distantPast

            do {
                let stream = engine.generate(prompt: prompt.text, addBOS: builder.addBOS, images: images,
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
                    // A decode cut off by a stop (the app went to the background) is a stop, not an error.
                    if stopRequested {
                        messages[i].stopped = true
                    } else {
                        messages[i].error = error.localizedDescription
                    }
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

    /// The app left the screen. iOS forbids GPU work in the background, so the
    /// reply stops here; a background task keeps the app alive until the token
    /// being decoded is done.
    func enterBackground() {
        speech.stop()
        voice.close()
        guard let generation, backgroundTask == .invalid else { return }
        stop()
        backgroundTask = UIApplication.shared.beginBackgroundTask { [weak self] in
            Task { @MainActor in self?.endBackgroundTask() }
        }
        Task {
            await generation.value
            endBackgroundTask()
        }
    }

    private func endBackgroundTask() {
        guard backgroundTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTask)
        backgroundTask = .invalid
    }

    func newChat() {
        stop()
        messages.removeAll()
    }
}

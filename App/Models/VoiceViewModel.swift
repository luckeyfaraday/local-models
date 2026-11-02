import AVFoundation
import Foundation
import Observation
import UIKit

/// A spoken conversation with the loaded model: tap to talk, tap again to
/// send. The model answers with speech and text from one generation (its
/// voice-chat mode), and both play and show as they are made. The user's
/// words are transcribed after each reply, for the screen and the next turn.
@Observable @MainActor
final class VoiceViewModel {
    enum Phase: Equatable {
        case idle
        case starting
        case listening
        case replying
        case failed(String)
    }

    struct Exchange: Identifiable, Equatable {
        let id = UUID()
        let number: Int
        let heardSeconds: Double
        /// What the user said, transcribed after the reply.
        var heard: String?
        var reply = ""
        var stats: VoiceStats?
        /// From the end of the user's turn to the first sound of the reply.
        var firstSound: Double?
        /// Times playback ran dry waiting for audio, and for how long in all.
        var gaps = 0
        var gapSeconds = 0.0
        var error: String?
    }

    private(set) var exchanges: [Exchange] = []
    private(set) var phase: Phase = .idle
    private(set) var transcribing = false

    var isBusy: Bool { phase == .starting || phase == .replying || transcribing }

    /// Speech held back before playback starts. While the model writes, text
    /// tokens and audio frames take turns (6 and 12): each block makes 0.96 s
    /// of speech in about 1.05 s on an iPhone 13 Pro, so playback falls ~0.09 s
    /// behind per block until the text is done. One block of audio up front
    /// covers replies of about 60 text tokens; with 0.6 s a 61-token reply ran
    /// dry 6 times.
    static let preroll = 0.96

    @ObservationIgnored private let engine: LlamaEngine
    @ObservationIgnored private let audio = VoiceAudio()
    @ObservationIgnored private var audioRunning = false
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var playback = Playback()
    /// Set by stop(): frames still in flight must not restart the sound.
    @ObservationIgnored private var muted = false
    @ObservationIgnored private var testIndex = 0

    private struct Playback {
        var rate = 24000
        var pending: [Float] = []
        var started = false
        var startedAt = Date.distantPast
        var ends = Date.distantPast
        var gaps = 0
        var gapSeconds = 0.0
        var all: [Float] = []
    }

    init(engine: LlamaEngine) {
        self.engine = engine
    }

    // MARK: - Actions

    /// The mic button: listen, send what was heard, or stop the reply.
    func talk(_ info: ModelInfo) {
        switch phase {
        case .listening:
            let samples = audio.endRecording(rate: info.audioInputRate)
            reply(to: samples, info: info, source: "mic")
        case .replying, .starting:
            stop()
        case .idle, .failed:
            listen(info)
        }
    }

    /// Debug hook: the next WAV in tmp/VoiceTest of the app's container
    /// stands in for the microphone. The button only shows when that folder has files.
    func sendTestFile(_ info: ModelInfo) {
        let files = Self.testFiles
        guard !files.isEmpty, phase != .listening, phase != .replying else { return }
        let file = files[testIndex % files.count]
        testIndex += 1
        phase = .starting
        Task {
            guard await prepare(info) else { return }
            do {
                let samples = try VoiceAudio.read(file, rate: info.audioInputRate)
                reply(to: samples, info: info, source: file.lastPathComponent)
            } catch {
                phase = .failed(error.localizedDescription)
            }
        }
    }

    /// Stops the reply being made and played.
    func stop() {
        muted = true
        if phase == .replying || phase == .starting { engine.cancel() }
        audio.silence()
        if phase == .listening { _ = audio.endRecording(rate: 16000) }
        if phase != .replying { phase = .idle }
    }

    func newConversation() {
        stop()
        task?.cancel()
        exchanges.removeAll()
        try? FileManager.default.removeItem(at: Self.outputFolder)
        phase = .idle
    }

    /// Leaving the screen: stop and give the audio session back.
    func close() {
        stop()
        if audioRunning {
            audio.stop()
            audioRunning = false
        }
    }

    static var testFolder: URL { FileManager.default.temporaryDirectory.appendingPathComponent("VoiceTest") }

    static var testFiles: [URL] {
        let all = (try? FileManager.default.contentsOfDirectory(at: testFolder, includingPropertiesForKeys: nil)) ?? []
        return all.filter { $0.pathExtension.lowercased() == "wav" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Each turn's question and reply as WAV files, plus log.txt with the numbers.
    static var outputFolder: URL { FileManager.default.temporaryDirectory.appendingPathComponent("Voice") }

    // MARK: - Turns

    private func listen(_ info: ModelInfo) {
        phase = .starting
        Task {
            guard await prepare(info) else { return }
            if AVAudioApplication.shared.recordPermission == .denied {
                phase = .failed("Microphone access is off. Turn it on in Settings › Local Models.")
                return
            }
            muted = true
            audio.beginRecording()
            phase = .listening
        }
    }

    private func prepare(_ info: ModelInfo) async -> Bool {
        guard !audioRunning else { return true }
        if AVAudioApplication.shared.recordPermission == .undetermined {
            _ = await AVAudioApplication.requestRecordPermission()
        }
        do {
            try audio.start(outputRate: info.audioOutputRate)
            audioRunning = true
            return true
        } catch {
            phase = .failed(error.localizedDescription)
            return false
        }
    }

    private func reply(to samples: [Float], info: ModelInfo, source: String) {
        let heardSeconds = Double(samples.count) / Double(info.audioInputRate)
        guard heardSeconds >= 0.3 else {
            phase = .idle
            return
        }
        let number = (exchanges.last?.number ?? 0) + 1
        let exchange = Exchange(number: number, heardSeconds: heardSeconds)
        let id = exchange.id
        exchanges.append(exchange)
        phase = .replying
        muted = false
        playback = Playback(rate: info.audioOutputRate)
        let turnStart = Date()
        let thermal = ProcessInfo.processInfo.thermalState
        UIApplication.shared.isIdleTimerDisabled = true

        // The previous turn may still be transcribing; its words go into this one.
        let previous = task
        task = Task {
            await previous?.value
            defer { UIApplication.shared.isIdleTimerDisabled = false }
            let history = exchanges.compactMap { e -> VoiceExchangeText? in
                guard e.id != id, let heard = e.heard, !heard.isEmpty, !e.reply.isEmpty else { return nil }
                return VoiceExchangeText(heard: heard, reply: e.reply)
            }
            do {
                for try await event in engine.voiceReply(history: history, audio: samples) {
                    switch event {
                    case .text(let piece):
                        update(id) { $0.reply += piece }
                    case .audio(let chunk, let rate):
                        received(chunk, rate: rate)
                    case .done(let stats):
                        update(id) { $0.stats = stats }
                    }
                }
                startPlayback()
            } catch {
                update(id) { $0.error = error.localizedDescription }
            }
            let firstSound = playback.started ? playback.startedAt.timeIntervalSince(turnStart) : nil
            let (gaps, gapSeconds) = (playback.gaps, playback.gapSeconds)
            update(id) {
                $0.reply = $0.reply.trimmingCharacters(in: .whitespacesAndNewlines)
                $0.firstSound = firstSound
                $0.gaps = gaps
                $0.gapSeconds = gapSeconds
            }
            if phase == .replying { phase = .idle }

            transcribing = true
            let heard = try? await engine.transcribe(samples)
            transcribing = false
            update(id) { $0.heard = heard ?? "" }
            save(id, question: samples, inputRate: info.audioInputRate, source: source, thermal: thermal)
        }
    }

    private func update(_ id: UUID, _ change: (inout Exchange) -> Void) {
        guard let i = exchanges.firstIndex(where: { $0.id == id }) else { return }
        change(&exchanges[i])
    }

    // MARK: - Playback

    private func received(_ chunk: [Float], rate: Int) {
        guard !muted else { return }
        playback.rate = rate
        playback.all += chunk
        if playback.started {
            schedule(chunk)
        } else {
            playback.pending += chunk
            if Double(playback.pending.count) / Double(rate) >= Self.preroll { startPlayback() }
        }
    }

    private func startPlayback() {
        guard !playback.started, !playback.pending.isEmpty, !muted else { return }
        let now = Date()
        playback.started = true
        playback.startedAt = now
        playback.ends = now
        schedule(playback.pending)
        playback.pending = []
    }

    private func schedule(_ chunk: [Float]) {
        let now = Date()
        let late = now.timeIntervalSince(playback.ends)
        if late > 0.02 {
            playback.gaps += 1
            playback.gapSeconds += late
        }
        playback.ends = max(playback.ends, now) + Double(chunk.count) / Double(playback.rate)
        audio.enqueue(chunk)
    }

    // MARK: - Files

    private func save(_ id: UUID, question: [Float], inputRate: Int, source: String, thermal: ProcessInfo.ThermalState) {
        guard let e = exchanges.first(where: { $0.id == id }) else { return }
        let folder = Self.outputFolder
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try? WAV.encode(question, sampleRate: inputRate).write(to: folder.appendingPathComponent("turn-\(e.number)-question.wav"))
        if !playback.all.isEmpty {
            try? WAV.encode(playback.all, sampleRate: playback.rate).write(to: folder.appendingPathComponent("turn-\(e.number)-reply.wav"))
        }
        let s = e.stats ?? VoiceStats()
        var line: [String: Any] = [
            "turn": e.number, "source": source, "heardSeconds": e.heardSeconds, "heard": e.heard ?? "", "reply": e.reply,
            "promptTokens": s.promptTokens, "audioRows": s.audioRows, "promptSeconds": s.promptSeconds,
            "textTokens": s.textTokens, "frames": s.frames, "generationSeconds": s.generationSeconds,
            "audioSeconds": s.audioSeconds, "realTimeFactor": s.realTimeFactor,
            "firstTextSeconds": s.firstTextSeconds ?? -1, "firstAudioSeconds": s.firstAudioSeconds ?? -1,
            "firstSoundSeconds": e.firstSound ?? -1, "preroll": Self.preroll, "gaps": e.gaps, "gapSeconds": e.gapSeconds,
            "stopped": s.stopped, "hitLimit": s.hitLimit, "thermal": thermal.rawValue,
        ]
        if let error = e.error { line["error"] = error }
        guard var data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
        data.append(0x0A)
        let log = folder.appendingPathComponent("log.txt")
        if let handle = try? FileHandle(forWritingTo: log) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: log)
        }
    }
}

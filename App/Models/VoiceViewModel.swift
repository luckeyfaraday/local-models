import AVFoundation
import Foundation
import Observation
import UIKit

/// A spoken conversation with the loaded model. Push to talk: tap to talk,
/// tap again to send. Hands-free: the mic stays open, a turn ends after a
/// short silence, and talking over a reply stops it. The model answers with
/// speech and text from one generation (its voice-chat mode), and both play
/// and show as they are made. The user's words are transcribed after each
/// reply, for the screen and the next turn.
@Observable @MainActor
final class VoiceViewModel {
    enum Phase: Equatable {
        case idle
        case starting
        case listening
        case replying
        case failed(String)
    }

    /// What a hands-free call is doing, for the screen.
    enum CallState: Equatable {
        case listening
        case hearing
        case thinking
        case speaking
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
        /// Talked over: only the part that was heard stays in the reply.
        var interrupted = false
        var playedSeconds: Double?
        var error: String?
    }

    private(set) var exchanges: [Exchange] = []
    private(set) var phase: Phase = .idle
    private(set) var transcribing = false
    /// Hands-free: the mic stays open and the detector takes the turns.
    private(set) var handsFree = false
    /// The detector hears speech.
    private(set) var hearing = false
    /// The reply is playing.
    private(set) var speaking = false

    var isBusy: Bool { phase == .starting || phase == .replying || transcribing }

    var callState: CallState {
        if hearing { return .hearing }
        if speaking { return .speaking }
        if phase == .replying { return .thinking }
        return .listening
    }

    /// Speech held back before playback starts. While the model writes, text
    /// tokens and audio frames take turns (6 and 12): each block makes 0.96 s
    /// of speech in about 1.05 s on an iPhone 13 Pro, so playback falls ~0.09 s
    /// behind per block until the text is done. One block of audio up front
    /// covers replies of about 60 text tokens; with 0.6 s a 61-token reply ran
    /// dry 6 times.
    static let preroll = 0.96

    /// Seconds of speech per text token in LFM3-Audio's replies (0.26–0.27 s
    /// measured on long ones): maps how much of an interrupted reply was heard
    /// to its text. The text runs ahead of the speech, so it can't be cut by
    /// what has been written.
    static let secondsPerToken = 0.27

    @ObservationIgnored private let engine: LlamaEngine
    @ObservationIgnored private let audio = VoiceAudio()
    @ObservationIgnored private var audioRunning = false
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var playback = Playback()
    /// The exchange whose reply may play: audio of an older one is dropped.
    @ObservationIgnored private var currentTurn: UUID?
    /// Set by stop(): frames still in flight must not restart the sound.
    @ObservationIgnored private var muted = false
    @ObservationIgnored private var handsFreeInfo: ModelInfo?
    @ObservationIgnored private var testIndex = 0

    private struct Playback {
        var rate = 24000
        var pending: [Float] = []
        var started = false
        var startedAt = Date.distantPast
        var ends = Date.distantPast
        var scheduledSeconds = 0.0
        var gaps = 0
        var gapSeconds = 0.0
        var all: [Float] = []
    }

    init(engine: LlamaEngine) {
        self.engine = engine
    }

    // MARK: - Actions

    /// The mic button: listen, send what was heard, or stop the reply. In
    /// hands-free it only stops the reply.
    func talk(_ info: ModelInfo) {
        if handsFree {
            if phase == .replying || speaking { interrupt() }
            return
        }
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

    func setHandsFree(_ on: Bool, info: ModelInfo) {
        guard on != handsFree else { return }
        if on {
            if phase == .listening {
                _ = audio.endRecording(rate: info.audioInputRate)
                phase = .idle
            }
            if case .failed = phase { phase = .idle }
            handsFree = true
            handsFreeInfo = info
            // Like a call: the screen stays on.
            UIApplication.shared.isIdleTimerDisabled = true
            Task {
                guard await prepare(info) else {
                    handsFree = false
                    return
                }
                guard handsFree else { return }
                // Debug builds of the test kit also keep the detector's levels.
                audio.tracing = !Self.testFiles.isEmpty
                audio.listen { [weak self] event in self?.heard(event) }
            }
        } else {
            handsFree = false
            hearing = false
            audio.listen(nil)
            if !isBusy { UIApplication.shared.isIdleTimerDisabled = false }
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
        speaking = false
        audio.replyPlaying = false
        if phase == .listening { _ = audio.endRecording(rate: 16000) }
        if phase != .replying { phase = .idle }
    }

    func newConversation() {
        stop()
        task?.cancel()
        currentTurn = nil
        exchanges.removeAll()
        try? FileManager.default.removeItem(at: Self.outputFolder)
        phase = .idle
    }

    /// Leaving the screen: stop, hang up and give the audio session back.
    func close() {
        if let info = handsFreeInfo { setHandsFree(false, info: info) }
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
            // A silent mic (no buffers yet) is restarted, at most twice.
            for _ in 0..<2 {
                try? await Task.sleep(nanoseconds: 400_000_000)
                if audio.hearsInput { break }
                try audio.restart()
            }
            return true
        } catch {
            phase = .failed(error.localizedDescription)
            return false
        }
    }

    /// The hands-free detector: a turn begins (or talks over the reply), or ends.
    private func heard(_ event: MicEvent) {
        guard handsFree, let info = handsFreeInfo else { return }
        switch event {
        case .speechStarted:
            hearing = true
            if phase == .replying || speaking { interrupt() }
        case .turnEnded(let samples, let rate):
            hearing = false
            reply(to: VoiceAudio.resample(samples, from: rate, to: Double(info.audioInputRate)), info: info, source: "hands-free")
        }
    }

    /// Talked over: stop generating and playing at once. The part that was
    /// heard stays in the conversation (the turn's task trims the text).
    private func interrupt() {
        guard let id = currentTurn else { return }
        let played = playedSeconds()
        update(id) {
            $0.interrupted = true
            $0.playedSeconds = played
        }
        muted = true
        if phase == .replying { engine.cancel() }
        audio.silence()
        speaking = false
        audio.replyPlaying = false
    }

    private func reply(to samples: [Float], info: ModelInfo, source: String) {
        let heardSeconds = Double(samples.count) / Double(info.audioInputRate)
        guard heardSeconds >= 0.3 else {
            if phase != .replying { phase = .idle }
            return
        }
        let number = (exchanges.last?.number ?? 0) + 1
        let exchange = Exchange(number: number, heardSeconds: heardSeconds)
        let id = exchange.id
        exchanges.append(exchange)
        currentTurn = id
        phase = .replying
        let turnStart = Date()
        let thermal = ProcessInfo.processInfo.thermalState
        let handsFreeTurn = handsFree
        UIApplication.shared.isIdleTimerDisabled = true

        // The previous turn may still be finishing (transcribing, trimming an
        // interrupted reply); its words go into this one.
        let previous = task
        task = Task {
            await previous?.value
            defer { UIApplication.shared.isIdleTimerDisabled = handsFree }
            guard currentTurn == id else { return }
            muted = false
            playback = Playback(rate: info.audioOutputRate)
            let history = exchanges.compactMap { e -> VoiceExchangeText? in
                guard e.id != id, let heard = e.heard, !heard.isEmpty, !e.reply.isEmpty else { return nil }
                return VoiceExchangeText(heard: heard, reply: e.reply)
            }
            var pieces: [String] = []
            do {
                for try await event in engine.voiceReply(history: history, audio: samples) {
                    switch event {
                    case .text(let piece):
                        pieces.append(piece)
                        update(id) { $0.reply += piece }
                    case .audio(let chunk, let rate):
                        received(chunk, rate: rate, turn: id)
                    case .done(let stats):
                        update(id) { $0.stats = stats }
                    }
                }
                startPlayback(turn: id)
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
            if phase == .replying && currentTurn == id { phase = .idle }
            let playing = Task { await finishPlayback(turn: id) }

            transcribing = true
            let heard = try? await engine.transcribe(samples)
            transcribing = false
            update(id) { $0.heard = heard ?? "" }

            let echo = await playing.value
            update(id) {
                if $0.interrupted, let played = $0.playedSeconds {
                    $0.reply = Self.heardPart(of: pieces, seconds: played)
                }
            }
            save(id, question: samples, inputRate: info.audioInputRate, source: source, thermal: thermal,
                 handsFree: handsFreeTurn, echo: echo)
        }
    }

    private func update(_ id: UUID, _ change: (inout Exchange) -> Void) {
        guard let i = exchanges.firstIndex(where: { $0.id == id }) else { return }
        change(&exchanges[i])
    }

    /// The text of the first `seconds` of a reply, to the last whole word.
    private static func heardPart(of pieces: [String], seconds: Double) -> String {
        let n = Int((seconds / secondsPerToken).rounded())
        guard n > 0 else { return "" }
        var text = pieces.prefix(n).joined()
        if n < pieces.count, let space = text.lastIndex(where: \.isWhitespace) {
            text = String(text[..<space])
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Playback

    private func received(_ chunk: [Float], rate: Int, turn: UUID) {
        guard !muted, turn == currentTurn else { return }
        playback.rate = rate
        playback.all += chunk
        if playback.started {
            schedule(chunk)
        } else {
            playback.pending += chunk
            if Double(playback.pending.count) / Double(rate) >= Self.preroll { startPlayback(turn: turn) }
        }
    }

    private func startPlayback(turn: UUID) {
        guard !playback.started, !playback.pending.isEmpty, !muted, turn == currentTurn else { return }
        let now = Date()
        playback.started = true
        playback.startedAt = now
        playback.ends = now
        speaking = true
        audio.replyPlaying = true
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
        let seconds = Double(chunk.count) / Double(playback.rate)
        playback.ends = max(playback.ends, now) + seconds
        playback.scheduledSeconds += seconds
        audio.enqueue(chunk)
    }

    /// How much of the current reply has been played.
    private func playedSeconds() -> Double {
        guard playback.started else { return 0 }
        return max(0, playback.scheduledSeconds - max(0, playback.ends.timeIntervalSinceNow))
    }

    /// Waits for the reply to finish playing (or be cut), then listens with
    /// the normal threshold again. Returns what the detector made of the echo.
    private func finishPlayback(turn: UUID) async -> EchoStats {
        while playback.started, !muted, currentTurn == turn {
            // A little past the end: the speaker still rings for a moment.
            let left = playback.ends.timeIntervalSinceNow + 0.3
            if left <= 0 { break }
            try? await Task.sleep(nanoseconds: UInt64(left * 1e9))
        }
        let echo = audio.echoStats()
        if currentTurn == turn {
            speaking = false
            audio.replyPlaying = false
        }
        return echo
    }

    // MARK: - Files

    private func save(_ id: UUID, question: [Float], inputRate: Int, source: String, thermal: ProcessInfo.ThermalState,
                      handsFree: Bool, echo: EchoStats) {
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
            "handsFree": handsFree, "interrupted": e.interrupted, "playedSeconds": e.playedSeconds ?? -1,
            "echoPeakDBFS": Double(echo.peak), "echoLongestRunSeconds": echo.longestRun,
            "noiseFloorDB": Double(echo.floor),
            "playbackStartEpoch": playback.started ? playback.startedAt.timeIntervalSince1970 : -1,
        ]
        if let error = e.error { line["error"] = error }
        if audio.tracing {
            let rows = audio.takeTrace().map { String(format: "%.3f,%.1f,%d", $0.time, $0.level, $0.echo ? 1 : 0) }
            try? (["time,db,echo"] + rows).joined(separator: "\n").write(
                to: folder.appendingPathComponent("turn-\(e.number)-levels.csv"), atomically: true, encoding: .utf8)
        }
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

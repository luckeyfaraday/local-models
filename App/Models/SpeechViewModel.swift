import Foundation
import Observation
import UIKit

/// Text to speech with the loaded model, plus the numbers that say whether the
/// phone keeps up: speech plays while it is generated, so it has to be made
/// faster than it is spoken.
@Observable @MainActor
final class SpeechViewModel {
    enum State: Equatable {
        case idle
        case speaking
        case done
        case failed(String)
    }

    private(set) var state: State = .idle
    private(set) var stats: SpeechStats?
    /// Audio generated so far in the current run, in seconds.
    private(set) var audioSoFar = 0.0
    /// Times playback ran dry waiting for audio, and for how long in total.
    private(set) var gaps = 0
    private(set) var gapSeconds = 0.0
    /// How much audio to buffer before playing for no gaps at all.
    private(set) var bufferNeeded = 0.0
    private(set) var thermalAtStart: ProcessInfo.ThermalState = .nominal

    @ObservationIgnored private let engine: LlamaEngine
    @ObservationIgnored private let player = SpeechPlayer()
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var samples: [Float] = []
    @ObservationIgnored private var sampleRate = 0
    /// Set by stop(): a frame still in flight must not restart the sound.
    @ObservationIgnored private var muted = false

    init(engine: LlamaEngine) {
        self.engine = engine
    }

    /// The last result as a WAV file, for sharing.
    private(set) var audioFile: URL?

    var isSpeaking: Bool { state == .speaking }
    var canReplay: Bool { !samples.isEmpty && !isSpeaking }

    func speak(_ request: SpeechRequest) {
        guard !isSpeaking, !request.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        state = .speaking
        stats = nil
        audioSoFar = 0
        gaps = 0
        gapSeconds = 0
        bufferNeeded = 0
        samples = []
        audioFile = nil
        muted = false
        thermalAtStart = ProcessInfo.processInfo.thermalState
        UIApplication.shared.isIdleTimerDisabled = true

        task = Task {
            defer {
                UIApplication.shared.isIdleTimerDisabled = false
                task = nil
            }
            // Where playback would be if every chunk were played the moment it arrived.
            var firstChunkAt: Date?
            var playbackEnds = Date.distantPast
            do {
                for try await event in engine.speak(request) {
                    switch event {
                    case .audio(let chunk, let rate):
                        guard !muted else { continue }
                        let now = Date()
                        if firstChunkAt == nil {
                            firstChunkAt = now
                            playbackEnds = now
                            sampleRate = rate
                            try player.start(sampleRate: rate)
                        }
                        let duration = Double(chunk.count) / Double(rate)
                        let late = now.timeIntervalSince(playbackEnds)
                        if late > 0.02 {
                            gaps += 1
                            gapSeconds += late
                        }
                        // Delay that would have hidden this chunk's lateness.
                        bufferNeeded = max(bufferNeeded, now.timeIntervalSince(firstChunkAt!) - audioSoFar)
                        playbackEnds = max(playbackEnds, now) + duration
                        audioSoFar += duration
                        samples += chunk
                        player.enqueue(chunk)
                    case .done(let result):
                        stats = result
                    }
                }
                state = .done
                audioFile = saveWAV()
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    /// Stops generating and playing.
    func stop() {
        muted = true
        if isSpeaking { engine.cancel() }
        player.stop()
    }

    private func saveWAV() -> URL? {
        guard !samples.isEmpty, sampleRate > 0 else { return nil }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("Local Models speech.wav")
        do {
            try WAV.encode(samples, sampleRate: sampleRate).write(to: url)
            return url
        } catch {
            return nil
        }
    }

    func replay() {
        guard canReplay else { return }
        muted = false
        do {
            try player.start(sampleRate: sampleRate)
            player.enqueue(samples)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}

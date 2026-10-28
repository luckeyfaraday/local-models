import AVFoundation

/// Plays generated speech as it arrives: each chunk is queued behind the last.
@MainActor
final class SpeechPlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private var format: AVAudioFormat?

    init() {
        engine.attach(node)
    }

    func start(sampleRate: Int) throws {
        node.stop()
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio)
        try session.setActive(true)
        if format?.sampleRate != Double(sampleRate) {
            guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1) else {
                throw PlayerError.badFormat(sampleRate)
            }
            engine.stop()
            engine.disconnectNodeOutput(node)
            engine.connect(node, to: engine.mainMixerNode, format: format)
            self.format = format
        }
        if !engine.isRunning { try engine.start() }
        node.play()
    }

    func enqueue(_ samples: [Float]) {
        guard let format, !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        node.scheduleBuffer(buffer)
    }

    func stop() {
        node.stop()
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    enum PlayerError: LocalizedError {
        case badFormat(Int)
        var errorDescription: String? {
            switch self {
            case .badFormat(let rate): return "Can't play audio at \(rate) Hz."
            }
        }
    }
}

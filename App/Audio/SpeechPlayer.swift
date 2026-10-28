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

/// 16-bit mono WAV, the format every app can open.
enum WAV {
    static func encode(_ samples: [Float], sampleRate: Int) -> Data {
        let dataBytes = samples.count * 2
        var d = Data(capacity: 44 + dataBytes)
        func put<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { d.append(contentsOf: $0) } }
        d.append(contentsOf: Array("RIFF".utf8)); put(UInt32(36 + dataBytes))
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); put(UInt32(16)); put(UInt16(1)); put(UInt16(1))
        put(UInt32(sampleRate)); put(UInt32(sampleRate * 2)); put(UInt16(2)); put(UInt16(16))
        d.append(contentsOf: Array("data".utf8)); put(UInt32(dataBytes))
        for x in samples { put(Int16((max(-1, min(1, x)) * 32767).rounded())) }
        return d
    }
}

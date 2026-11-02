import AVFoundation

/// Microphone and speaker of a spoken conversation, on one engine so the
/// iPhone's voice processing can cancel the reply's echo from the mic.
/// Not tied to the main actor: the tap runs on the audio thread.
final class VoiceAudio: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let sink = Sink()
    private var outputRate = 0
    private var configured = false

    /// Opens the session and starts the engine. The reply plays at `outputRate`.
    func start(outputRate: Int) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker])
        try session.setActive(true)
        if !configured {
            // Echo cancellation. Without it the mic still works, only less cleanly.
            try? engine.inputNode.setVoiceProcessingEnabled(true)
            let input = engine.inputNode.outputFormat(forBus: 0)
            guard input.sampleRate > 0, input.channelCount > 0 else { throw VoiceAudioError.noMicrophone }
            guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(outputRate), channels: 1) else {
                throw VoiceAudioError.badFormat(outputRate)
            }
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            sink.rate = input.sampleRate
            engine.inputNode.installTap(onBus: 0, bufferSize: 4096, format: input, block: sink.tap())
            self.outputRate = outputRate
            configured = true
        }
        if !engine.isRunning {
            engine.prepare()
            try engine.start()
        }
        player.play()
    }

    func stop() {
        sink.recording = false
        player.stop()
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    // MARK: Recording

    func beginRecording() {
        silence()
        sink.begin()
    }

    /// What the mic heard since beginRecording(), mono at `rate` Hz.
    func endRecording(rate: Int) -> [Float] {
        let (samples, from) = sink.end()
        return Self.resample(samples, from: from, to: Double(rate))
    }

    // MARK: Playback

    func enqueue(_ samples: [Float]) {
        guard configured, !samples.isEmpty,
              let format = AVAudioFormat(standardFormatWithSampleRate: Double(outputRate), channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = buffer.frameCapacity
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        guard engine.isRunning else { return }
        if !player.isPlaying { player.play() }
        player.scheduleBuffer(buffer)
    }

    /// Drops the audio still queued, the engine keeps running.
    func silence() {
        guard engine.isRunning else { return }
        player.stop()
        player.play()
    }

    // MARK: Files and rates

    /// A WAV (or any file AVAudioFile reads) as mono samples at `rate` Hz.
    static func read(_ url: URL, rate: Int) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw VoiceAudioError.unreadable
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else { throw VoiceAudioError.unreadable }
        let samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
        return resample(samples, from: format.sampleRate, to: Double(rate))
    }

    static func resample(_ samples: [Float], from: Double, to: Double) -> [Float] {
        guard from != to, !samples.isEmpty,
              let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: from, channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: to, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat),
              let input = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(samples.count)),
              let output = AVAudioPCMBuffer(pcmFormat: outFormat,
                                            frameCapacity: AVAudioFrameCount(Double(samples.count) * to / from) + 4096)
        else { return samples }
        input.frameLength = input.frameCapacity
        samples.withUnsafeBufferPointer { input.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        let fed = Once()
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if fed.done {
                inputStatus.pointee = .endOfStream
                return nil
            }
            fed.done = true
            inputStatus.pointee = .haveData
            return input
        }
        guard status != .error, let channel = output.floatChannelData?[0] else { return samples }
        return Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
    }

    private final class Once: @unchecked Sendable {
        var done = false
    }

    /// Collects mic samples on the audio thread while recording.
    private final class Sink: @unchecked Sendable {
        private let lock = NSLock()
        private var samples: [Float] = []
        private var on = false
        var rate = 48000.0

        var recording: Bool {
            get { lock.lock(); defer { lock.unlock() }; return on }
            set { lock.lock(); on = newValue; lock.unlock() }
        }

        func begin() {
            lock.lock()
            samples.removeAll(keepingCapacity: true)
            on = true
            lock.unlock()
        }

        func end() -> ([Float], Double) {
            lock.lock()
            defer { lock.unlock() }
            on = false
            return (samples, rate)
        }

        func tap() -> AVAudioNodeTapBlock {
            return { [self] buffer, _ in
                guard let channel = buffer.floatChannelData?[0] else { return }
                let n = Int(buffer.frameLength)
                lock.lock()
                if on { samples.append(contentsOf: UnsafeBufferPointer(start: channel, count: n)) }
                lock.unlock()
            }
        }
    }
}

enum VoiceAudioError: LocalizedError {
    case noMicrophone
    case badFormat(Int)
    case unreadable

    var errorDescription: String? {
        switch self {
        case .noMicrophone: return "No microphone is available."
        case .badFormat(let rate): return "Can't play audio at \(rate) Hz."
        case .unreadable: return "Couldn't read the audio file."
        }
    }
}

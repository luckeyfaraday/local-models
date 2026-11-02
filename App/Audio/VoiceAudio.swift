import Accelerate
import AVFoundation

/// What the hands-free detector heard, delivered on the main actor.
enum MicEvent: Sendable {
    /// Speech went on for VoiceDetector.startSeconds: a turn (or an interruption) has begun.
    case speechStarted
    /// The turn ended in silence: its audio, mono at `rate` Hz, from just before it began.
    case turnEnded([Float], rate: Double)
}

/// The detector's view of the reply's echo, for the log.
struct EchoStats: Equatable {
    var peakAboveFloor: Float = -100
    var longestRun = 0.0
    var floor: Float = 0
}

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
        sink.setHandsFree(nil)
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

    // MARK: Hands-free

    /// With a handler, the mic stays open and the detector splits it into
    /// turns; nil turns it off. Only level math runs per buffer, never the model.
    func listen(_ handler: (@MainActor @Sendable (MicEvent) -> Void)?) {
        sink.setHandsFree(handler)
    }

    /// While the reply plays, speech has to be louder to count (echo).
    var replyPlaying: Bool {
        get { sink.echo }
        set { sink.echo = newValue }
    }

    /// The echo numbers since the reply started playing.
    func echoStats() -> EchoStats { sink.echoStats() }

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

    /// Mic samples on the tap thread: everything while recording (push to
    /// talk), or turns cut by the detector (hands-free).
    private final class Sink: @unchecked Sendable {
        private let lock = NSLock()
        private var samples: [Float] = []
        private var on = false
        var rate = 48000.0

        private var handler: (@MainActor @Sendable (MicEvent) -> Void)?
        private var detector = VoiceDetector()
        private var echoOn = false
        /// The last moments before speech starts, so the turn keeps its first syllable.
        private var ring: [Float] = []
        private var turn: [Float] = []
        private var frame: [Float] = []
        private static let prerollSeconds = 0.6
        /// Silence kept at the end of a turn.
        private static let tailSeconds = 0.3

        var recording: Bool {
            get { lock.lock(); defer { lock.unlock() }; return on }
            set { lock.lock(); on = newValue; lock.unlock() }
        }

        var echo: Bool {
            get { lock.lock(); defer { lock.unlock() }; return echoOn }
            set {
                lock.lock()
                if newValue && !echoOn { detector.resetEchoStats() }
                echoOn = newValue
                lock.unlock()
            }
        }

        func setHandsFree(_ handler: (@MainActor @Sendable (MicEvent) -> Void)?) {
            lock.lock()
            self.handler = handler
            detector = VoiceDetector()
            ring.removeAll()
            turn.removeAll()
            frame.removeAll()
            lock.unlock()
        }

        func echoStats() -> EchoStats {
            lock.lock()
            defer { lock.unlock() }
            return EchoStats(peakAboveFloor: detector.echoPeak, longestRun: detector.echoLongestRun, floor: detector.floor)
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
                let input = UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength))
                lock.lock()
                if on { samples.append(contentsOf: input) }
                var events: [MicEvent] = []
                let handler = self.handler
                if handler != nil { detect(input, into: &events) }
                lock.unlock()
                if let handler {
                    for event in events { Task { @MainActor in handler(event) } }
                }
            }
        }

        /// Runs under the lock: 20 ms frames through the detector.
        private func detect(_ input: UnsafeBufferPointer<Float>, into events: inout [MicEvent]) {
            let frameLength = max(1, Int(rate * VoiceDetector.frameSeconds))
            var i = 0
            while i < input.count {
                let n = min(frameLength - frame.count, input.count - i)
                frame.append(contentsOf: UnsafeBufferPointer(rebasing: input[i..<(i + n)]))
                i += n
                guard frame.count == frameLength else { break }

                var meanSquare: Float = 0
                vDSP_measqv(frame, 1, &meanSquare, vDSP_Length(frameLength))
                let db = 10 * log10(meanSquare + 1e-12)
                let wasInSpeech = detector.inSpeech
                let event = detector.process(level: db, echo: echoOn)
                if wasInSpeech || event == .started {
                    if event == .started {
                        // The frames that made it speech are in the ring, with the pre-roll before them.
                        turn = ring
                        ring.removeAll(keepingCapacity: true)
                    }
                    turn.append(contentsOf: frame)
                } else {
                    ring.append(contentsOf: frame)
                }
                if event == .started { events.append(.speechStarted) }
                if event == .ended {
                    let drop = Int(rate * (VoiceDetector.endSeconds - Self.tailSeconds))
                    let audio = Array(turn.dropLast(min(drop, turn.count)))
                    turn.removeAll(keepingCapacity: true)
                    events.append(.turnEnded(audio, rate: rate))
                }
                frame.removeAll(keepingCapacity: true)
            }
            let keep = Int(rate * (Self.prerollSeconds + VoiceDetector.startSeconds + VoiceDetector.blipGapSeconds))
            if ring.count > keep { ring.removeFirst(ring.count - keep) }
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

import Foundation

/// Voice activity from mic levels alone: speech is a level well above an
/// adaptive noise floor. It runs on every 20 ms of mic audio while a reply
/// plays, so it has to cost next to nothing: one mean square per frame.
struct VoiceDetector {
    enum Event: Equatable {
        case started
        case ended
    }

    static let frameSeconds = 0.02
    /// Speech shorter than this is a blip, not a turn.
    static let startSeconds = 0.3
    /// Silence after speech that ends the turn.
    static let endSeconds = 0.7
    /// A pause inside a word or between words doesn't reset the start count.
    static let blipGapSeconds = 0.15
    static let maxTurnSeconds = 30.0
    /// The floor is learned from the first half second.
    static let calibrationSeconds = 0.5

    /// dB above the noise floor that count as speech.
    static let margin: Float = 10
    /// Extra while the reply plays, so what voice processing leaves of its echo doesn't count.
    static let echoMargin: Float = 8
    /// Quieter than this (dBFS) is never speech. Voice processing suppresses
    /// noise to about -100 dBFS between sounds, so this is usually the threshold.
    static let minLevel: Float = -55
    /// The same while the reply plays: its echo leaks through in short bursts.
    static let echoMinLevel: Float = -40
    /// Over the reply, voice processing chops the user's speech into bursts
    /// (double talk): gaps up to 60 ms (3 frames) don't break it. Measured on
    /// an iPhone 13 Pro: a question played over a reply got 0.26 s unbroken
    /// above -40 dBFS, 0.52 s with these gaps; 2 minutes of replies with room
    /// sound never got 0.3 s.
    static let echoGapSeconds = 0.07

    private(set) var floor: Float = -60
    private(set) var inSpeech = false
    private var calibrated = 0.0
    private var calibrationSum: Float = 0
    private var voiced = 0.0
    private var quiet = 0.0
    private var turn = 0.0

    /// While the reply plays: the loudest frame (dBFS), and the longest
    /// unbroken run over the interrupt threshold (it fires at startSeconds).
    private(set) var echoPeak: Float = -100
    private(set) var echoLongestRun = 0.0
    private var echoRun = 0.0

    mutating func resetEchoStats() {
        echoPeak = -100
        echoLongestRun = 0
        echoRun = 0
    }

    /// One frame's level in dBFS. `echo` is true while the reply plays.
    mutating func process(level db: Float, echo: Bool) -> Event? {
        let frame = Self.frameSeconds
        if calibrated < Self.calibrationSeconds {
            calibrationSum += db
            calibrated += frame
            floor = calibrationSum / Float(calibrated / frame)
            return nil
        }
        let threshold = echo
            ? max(floor + Self.margin + Self.echoMargin, Self.echoMinLevel)
            : max(floor + Self.margin, Self.minLevel)
        let loud = db > threshold
        if echo {
            echoPeak = max(echoPeak, db)
            echoRun = loud ? echoRun + frame : 0
            echoLongestRun = max(echoLongestRun, echoRun)
        }

        if !inSpeech {
            if loud {
                voiced += frame
                quiet = 0
            } else {
                quiet += frame
                if echo ? quiet > Self.echoGapSeconds : quiet >= Self.blipGapSeconds { voiced = 0 }
                // Down fast, up slowly (about 1 dB/s), only from frames that aren't speech.
                floor += (db < floor ? 0.2 : 0.005) * (db - floor)
            }
            if voiced >= Self.startSeconds {
                inSpeech = true
                turn = voiced
                quiet = 0
                return .started
            }
        } else {
            turn += frame
            quiet = loud ? 0 : quiet + frame
            if quiet >= Self.endSeconds || turn >= Self.maxTurnSeconds {
                inSpeech = false
                voiced = 0
                quiet = 0
                return .ended
            }
        }
        return nil
    }
}

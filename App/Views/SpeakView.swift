import SwiftUI

/// Type something, hear the loaded model say it, and see whether the phone
/// makes speech faster than it is spoken.
struct SpeakView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(\.dismiss) private var dismiss
    @AppStorage(SettingKey.speechVoice) private var voice = "female"
    @AppStorage(SettingKey.speechLanguage) private var language = "en"
    @AppStorage(SettingKey.speechText) private var text = SpeechRequest.sample
    @FocusState private var editing: Bool

    private var speech: SpeechViewModel { chat.speech }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Text to speak", text: $text, axis: .vertical)
                        .lineLimit(3...10)
                        .focused($editing)
                }
                Section {
                    Picker("Voice", selection: $voice) {
                        Text("Female").tag("female")
                        Text("Male").tag("male")
                    }
                    .pickerStyle(.segmented)
                    Picker("Language", selection: $language) {
                        ForEach(SpeechLanguage.all) { Text($0.name).tag($0.code) }
                    }
                }
                Section {
                    if speech.isSpeaking {
                        Button(role: .destructive) { speech.stop() } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                    } else {
                        Button {
                            editing = false
                            speech.speak(SpeechRequest(text: text, voice: voice, language: language))
                        } label: {
                            Label("Speak", systemImage: "waveform")
                        }
                        .disabled(chat.isBusy || chat.info?.speaks != true
                                  || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    if speech.canReplay {
                        Button { speech.replay() } label: { Label("Play again", systemImage: "play.fill") }
                        if let file = speech.audioFile {
                            ShareLink(item: file) { Label("Share audio", systemImage: "square.and.arrow.up") }
                        }
                    }
                }
                results
            }
            .scrollDismissesKeyboard(.immediately)
            .navigationTitle("Speak")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                // The keyboard covers the Speak button, and return adds a new line.
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { editing = false }
                }
            }
            .onDisappear { speech.stop() }
        }
    }

    @ViewBuilder
    private var results: some View {
        switch speech.state {
        case .idle:
            EmptyView()
        case .speaking:
            Section("Speaking") {
                LabeledContent("Audio so far", value: seconds(speech.audioSoFar))
            }
        case .failed(let message):
            Section {
                Text(message).foregroundStyle(.orange)
            }
        case .done:
            if let s = speech.stats {
                Section {
                    LabeledContent("Speed") {
                        Text(String(format: "%.2f× real time", s.realTimeFactor))
                            .foregroundStyle(s.realTimeFactor >= 1 ? Color.green : Color.orange)
                    }
                    LabeledContent("Audio", value: "\(seconds(s.audioSeconds)) made in \(seconds(s.totalSeconds))")
                    LabeledContent("First sound after", value: s.firstAudioSeconds.map(seconds) ?? "–")
                    LabeledContent("Frames", value: String(format: "%ld at %.1f/s (speech needs 12.5/s)", s.frames, s.framesPerSecond))
                    LabeledContent("Text read", value: "\(s.promptTokens) tokens in \(seconds(s.promptSeconds))"
                                   + (s.chunks > 1 ? ", \(s.chunks) pieces" : ""))
                    LabeledContent("Gaps while playing", value: speech.gaps == 0 ? "none" : "\(speech.gaps), \(seconds(speech.gapSeconds)) in all")
                    LabeledContent("Buffer for no gaps", value: seconds(speech.bufferNeeded))
                    LabeledContent("Phone temperature", value: thermal(speech.thermalAtStart))
                } header: {
                    Text(s.stopped ? "Stopped" : "Result")
                } footer: {
                    Text(s.hitFrameLimit
                         ? "Stopped at the two-minute limit."
                         : "Above 1× the phone makes speech faster than it plays, so replies can be spoken as they are written.")
                }
            }
        }
    }

    private func seconds(_ value: Double) -> String {
        String(format: "%.2f s", value)
    }

    private func thermal(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "normal"
        case .fair: return "warm"
        case .serious: return "hot"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

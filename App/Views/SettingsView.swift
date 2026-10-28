import SwiftUI

struct SettingsView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(\.dismiss) private var dismiss
    @AppStorage(SettingKey.thinking) private var thinking = false
    @AppStorage(SettingKey.contextLength) private var contextLength = 8192
    @AppStorage(SettingKey.preset) private var preset = SamplingPreset.general.rawValue
    @AppStorage(SettingKey.maxTokens) private var maxTokens = 4096
    @AppStorage(SettingKey.systemPrompt) private var systemPrompt = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Think before answering", isOn: $thinking)
                } footer: {
                    Text(chat.supportsThinking || chat.info == nil
                         ? "Reasoning models write their thoughts first. Better on hard problems, but a small model can think for minutes before it answers."
                         : "The loaded model doesn't have a reasoning mode.")
                }

                Section("Sampling") {
                    Picker("Preset", selection: $preset) {
                        ForEach(SamplingPreset.allCases) { p in
                            VStack(alignment: .leading) {
                                Text(p.label)
                                Text(p.detail).font(.caption).foregroundStyle(.secondary)
                            }
                            .tag(p.rawValue)
                        }
                    }
                    .pickerStyle(.inline)
                    .labelsHidden()
                    Stepper("Max reply: \(maxTokens) tokens", value: $maxTokens, in: 256...16384, step: 256)
                }

                Section {
                    Picker("Context length", selection: $contextLength) {
                        ForEach(AppSettings.contextChoices, id: \.self) { n in
                            Text("\(n / 1024)K tokens").tag(n)
                        }
                    }
                } footer: {
                    Text("How much of the chat the model can see. Longer uses more memory. Changing it reloads the model on your next message.")
                }

                Section("System prompt") {
                    TextField("Optional instructions for every chat", text: $systemPrompt, axis: .vertical)
                        .lineLimit(2...8)
                }

                Section("About") {
                    if let info = chat.info {
                        LabeledContent("Model", value: info.name)
                        LabeledContent("Architecture", value: info.architecture)
                        LabeledContent("Size", value: "\(Format.params(info.parameterCount)) · \(Format.bytes(Int64(info.sizeBytes)))")
                        LabeledContent("Trained context", value: "\(info.trainContext / 1024)K")
                    }
                    LabeledContent("llama.cpp", value: Bundle.main.object(forInfoDictionaryKey: "LlamaBuild") as? String ?? "?")
                    LabeledContent("App", value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}

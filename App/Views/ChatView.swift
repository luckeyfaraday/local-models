import SwiftUI

struct ChatView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ModelStore.self) private var store
    @Environment(\.scenePhase) private var scenePhase
    @State private var input = ""
    @State private var showModels = false
    @State private var showSettings = false
    @State private var showSpeak = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if chat.messages.isEmpty {
                    EmptyChatView(showModels: $showModels)
                } else {
                    messageList
                }
                inputBar
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3") }
                        .accessibilityLabel("Settings")
                }
                ToolbarItem(placement: .principal) {
                    Button { showModels = true } label: { modelTitle }
                        .accessibilityLabel("Models")
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    if chat.info?.speaks == true {
                        Button { showSpeak = true } label: { Image(systemName: "waveform") }
                            .disabled(chat.isGenerating)
                            .accessibilityLabel("Speak")
                    }
                    Button { chat.newChat() } label: { Image(systemName: "square.and.pencil") }
                        .disabled(chat.messages.isEmpty)
                        .accessibilityLabel("New chat")
                }
            }
            .sheet(isPresented: $showModels) { ModelsView() }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .sheet(isPresented: $showSpeak) { SpeakView() }
        }
        .task { await chat.autoload(from: store) }
        .onChange(of: scenePhase) { _, phase in
            // iOS forbids GPU work in the background, so stop before the app leaves the screen.
            if phase != .active {
                chat.stop()
                chat.speech.stop()
            }
        }
    }

    private var modelTitle: some View {
        HStack(spacing: 4) {
            Text(titleText).font(.headline).lineLimit(1)
            Image(systemName: "chevron.down").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
        }
        .foregroundStyle(.primary)
    }

    private var titleText: String {
        switch chat.status {
        case .loading: return "Loading…"
        default: return chat.info?.name ?? "Choose a model"
        }
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                // Not lazy: a reasoning block can be dozens of screens tall, which
                // breaks LazyVStack's height estimate and scrolls into blank space.
                VStack(alignment: .leading, spacing: 18) {
                    ForEach(chat.messages) { message in
                        MessageRow(message: message, isStreaming: chat.isGenerating && message.id == chat.messages.last?.id)
                            .id(message.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .defaultScrollAnchor(.bottom)
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: chat.messages.last?.raw) { _, _ in
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            .onChange(of: chat.messages.count) { _, _ in
                withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }

    private var canSend: Bool {
        chat.status == .ready && !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField(placeholder, text: $input, axis: .vertical)
                .lineLimit(1...6)
                .focused($inputFocused)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .disabled(chat.loadedFile == nil)

            if chat.isGenerating {
                Button { chat.stop() } label: {
                    Image(systemName: "stop.circle.fill").font(.system(size: 34))
                }
                .accessibilityLabel("Stop")
            } else {
                Button(action: send) {
                    Image(systemName: "arrow.up.circle.fill").font(.system(size: 34))
                }
                .disabled(!canSend)
                .accessibilityLabel("Send")
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 8)
        .background(.bar)
    }

    private var placeholder: String {
        chat.loadedFile == nil ? "Load a model to start" : "Message"
    }

    private func send() {
        guard canSend else { return }
        chat.send(input, store: store)
        input = ""
    }
}

private struct EmptyChatView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ModelStore.self) private var store
    @Binding var showModels: Bool

    var body: some View {
        VStack(spacing: 14) {
            Spacer()
            switch chat.status {
            case .loading(let name):
                ProgressView()
                Text("Loading \(name)…").foregroundStyle(.secondary)
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
                Text(message).multilineTextAlignment(.center).foregroundStyle(.secondary)
                Button("Choose a model") { showModels = true }.buttonStyle(.borderedProminent)
            case .noModel:
                Image(systemName: "cpu").font(.system(size: 44)).foregroundStyle(.tint)
                if store.models.isEmpty {
                    Text("No models on this iPhone yet").font(.title3.weight(.semibold))
                    Text("Download a GGUF from Hugging Face. It runs entirely on the phone.")
                        .multilineTextAlignment(.center).foregroundStyle(.secondary)
                    Button("Get a model") { showModels = true }.buttonStyle(.borderedProminent)
                } else {
                    Text("Pick a model to load").font(.title3.weight(.semibold))
                    Button("Choose a model") { showModels = true }.buttonStyle(.borderedProminent)
                }
            case .ready, .generating:
                if let info = chat.info {
                    Image(systemName: "cpu").font(.system(size: 44)).foregroundStyle(.tint)
                    Text(info.name).font(.title3.weight(.semibold))
                    Text("\(Format.params(info.parameterCount)) · \(info.architecture) · \(info.contextLength / 1024)K context")
                        .font(.footnote).foregroundStyle(.secondary)
                    if info.speaks {
                        Label("Can speak: tap the waveform button", systemImage: "waveform")
                            .font(.footnote).foregroundStyle(.tint)
                    }
                    Text("Runs on this iPhone. Nothing leaves the device.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .padding(32)
        .frame(maxWidth: .infinity)
    }
}

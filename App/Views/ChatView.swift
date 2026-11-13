import PhotosUI
import SwiftUI

struct ChatView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(ModelStore.self) private var store
    @Environment(\.scenePhase) private var scenePhase
    @State private var input = ""
    @State private var showModels = false
    @State private var showSettings = false
    @State private var showSpeak = false
    @State private var showVoice = false
    @State private var attachment: ChatImage?
    @State private var preparingImage = false
    @State private var showPhotos = false
    @State private var photoItem: PhotosPickerItem?
    @State private var imageError: String?
    /// The chat follows a streaming reply only while its end is on screen.
    @State private var atBottom = true
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if chat.messages.isEmpty {
                    EmptyChatView(showModels: $showModels)
                } else {
                    messageList
                    if case .failed(let message) = chat.status { failureBanner(message) }
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
                    if chat.info?.talks == true {
                        Button { showVoice = true } label: { Image(systemName: "mic") }
                            .disabled(chat.isGenerating)
                            .accessibilityLabel("Talk")
                    }
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
            .sheet(isPresented: $showVoice) { VoiceView() }
            .photosPicker(isPresented: $showPhotos, selection: $photoItem, matching: .images)
            .onChange(of: photoItem) { _, item in
                guard let item else { return }
                photoItem = nil
                attach {
                    guard let data = try await item.loadTransferable(type: Data.self) else { return nil }
                    return await ChatImage.make(from: data)
                }
            }
            .onChange(of: chat.seesImages) { _, sees in
                if !sees { attachment = nil }
            }
        }
        .task { await chat.autoload(from: store) }
        .onChange(of: scenePhase) { _, phase in
            // Inactive (Control Center, an alert, the app switcher) leaves a reply
            // running. The background doesn't allow GPU work or an open mic.
            if phase == .background { chat.enterBackground() }
            if phase == .active { chat.voice.checkAudio() }
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

    /// A load that failed with a chat on screen (the context length changed, say).
    private func failureBanner(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
            Text(message).font(.footnote).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("Models") { showModels = true }.font(.footnote.weight(.semibold))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(.secondarySystemBackground))
    }

    private var messageList: some View {
        GeometryReader { viewport in
            ScrollViewReader { proxy in
                messageScroll(proxy: proxy, height: viewport.size.height)
            }
        }
    }

    private func messageScroll(proxy: ScrollViewProxy, height: CGFloat) -> some View {
        ScrollView {
            // Not lazy: a reasoning block can be dozens of screens tall, which
            // breaks LazyVStack's height estimate and scrolls into blank space.
            VStack(alignment: .leading, spacing: 18) {
                ForEach(chat.messages) { message in
                    MessageRow(message: message, isStreaming: chat.isGenerating && message.id == chat.messages.last?.id)
                        .id(message.id)
                }
                Color.clear.frame(height: 1).id("bottom")
                    .background {
                        GeometryReader { marker in
                            // Within a few lines of the end counts: one flush of a
                            // streaming reply adds less than that.
                            let near = marker.frame(in: .named("chat")).maxY <= height + 80
                            Color.clear.onChange(of: near, initial: true) { _, near in atBottom = near }
                        }
                    }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .coordinateSpace(.named("chat"))
        .defaultScrollAnchor(.bottom)
        .scrollDismissesKeyboard(.interactively)
        .onChange(of: chat.messages.last?.raw) { _, _ in
            // Scrolled up to read: the reply keeps streaming below, out of the way.
            if atBottom { proxy.scrollTo("bottom", anchor: .bottom) }
        }
        .onChange(of: chat.messages.count) { _, _ in
            withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
        }
    }

    private var canSend: Bool {
        chat.status == .ready && !preparingImage
            && (attachment != nil || !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private var inputBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            if attachment != nil || preparingImage || imageError != nil {
                attachmentPreview
            }
            HStack(alignment: .bottom, spacing: 10) {
                if chat.seesImages { attachMenu }
                textField
                sendButton
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 8)
        .background(.bar)
    }

    private var attachMenu: some View {
        Menu {
            Button { showPhotos = true } label: { Label("Photo Library", systemImage: "photo.on.rectangle") }
            Button {
                attach {
                    guard let image = UIPasteboard.general.image else { return nil }
                    return await ChatImage.make(from: image)
                }
            } label: {
                Label("Paste Image", systemImage: "doc.on.clipboard")
            }
        } label: {
            Image(systemName: "photo.badge.plus")
                .font(.system(size: 22))
                .frame(width: 34, height: 40)
        }
        .disabled(chat.isBusy)
        .accessibilityLabel("Attach image")
    }

    private var attachmentPreview: some View {
        HStack(spacing: 8) {
            if let attachment {
                Image(uiImage: attachment.thumbnail)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 64, height: 64)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .overlay(alignment: .topTrailing) {
                        Button { self.attachment = nil } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 20))
                                .symbolRenderingMode(.palette)
                                .foregroundStyle(.white, .black.opacity(0.6))
                        }
                        .offset(x: 6, y: -6)
                        .accessibilityLabel("Remove image")
                    }
                    .accessibilityLabel("Attached image")
            } else if preparingImage {
                ProgressView()
                    .frame(width: 64, height: 64)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
            if let imageError {
                Text(imageError).font(.footnote).foregroundStyle(.orange)
            }
        }
        .padding(.top, 6)
        .padding(.leading, chat.seesImages ? 44 : 0)
    }

    private func attach(_ load: @escaping () async throws -> ChatImage?) {
        imageError = nil
        preparingImage = true
        Task {
            defer { preparingImage = false }
            do {
                if let image = try await load() {
                    attachment = image
                } else {
                    imageError = "No image to attach."
                }
            } catch {
                imageError = "Couldn't read that image: \(error.localizedDescription)"
            }
        }
    }

    private var textField: some View {
        TextField(placeholder, text: $input, axis: .vertical)
            .lineLimit(1...6)
            .focused($inputFocused)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .disabled(chat.loadedFile == nil)
    }

    @ViewBuilder private var sendButton: some View {
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

    private var placeholder: String {
        if chat.loadedFile == nil { return "Load a model to start" }
        return attachment != nil ? "Ask about the image" : "Message"
    }

    private func send() {
        guard canSend else { return }
        chat.send(input, images: attachment.map { [$0] } ?? [], store: store)
        input = ""
        attachment = nil
        imageError = nil
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
                    if info.talks {
                        Label("Can talk: tap the microphone button", systemImage: "mic")
                            .font(.footnote).foregroundStyle(.tint)
                    } else if info.speaks {
                        Label("Can speak: tap the waveform button", systemImage: "waveform")
                            .font(.footnote).foregroundStyle(.tint)
                    }
                    if info.seesImages {
                        Label("Can see images: attach a photo or screenshot", systemImage: "photo")
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

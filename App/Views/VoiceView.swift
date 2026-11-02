import SwiftUI

/// Talk to the loaded model: tap the mic, speak, tap again. The model answers
/// out loud, with its text on screen, and the conversation carries on.
struct VoiceView: View {
    @Environment(ChatViewModel.self) private var chat
    @Environment(\.dismiss) private var dismiss
    @State private var hasTestFiles = false

    private var voice: VoiceViewModel { chat.voice }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if voice.exchanges.isEmpty {
                    empty
                } else {
                    transcript
                }
                controls
            }
            .navigationTitle("Talk")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItemGroup(placement: .topBarLeading) {
                    Button { voice.newConversation() } label: { Image(systemName: "square.and.pencil") }
                        .disabled(voice.exchanges.isEmpty)
                        .accessibilityLabel("New conversation")
                    if hasTestFiles, let info = chat.info {
                        Button { voice.sendTestFile(info) } label: { Image(systemName: "testtube.2") }
                            .disabled(voice.isBusy || voice.phase == .listening)
                            .accessibilityLabel("Test question")
                    }
                }
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .onAppear { hasTestFiles = !VoiceViewModel.testFiles.isEmpty }
            .onDisappear { voice.close() }
        }
    }

    private var empty: some View {
        VStack(spacing: 12) {
            Spacer()
            Image(systemName: "bubble.left.and.text.bubble.right").font(.system(size: 44)).foregroundStyle(.tint)
            Text("Talk with \(chat.info?.name ?? "the model")").font(.title3.weight(.semibold))
            Text("Tap the microphone, speak, and tap again. Or turn on hands-free and just talk; speak over a reply to interrupt it. English only.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(32)
        .frame(maxWidth: .infinity)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    ForEach(voice.exchanges) { exchange in
                        ExchangeRow(exchange: exchange,
                                    isReplying: voice.phase == .replying && exchange.id == voice.exchanges.last?.id)
                            .id(exchange.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .defaultScrollAnchor(.bottom)
            .onChange(of: voice.exchanges.last?.reply) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: voice.exchanges.count) { _, _ in
                withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }

    private var controls: some View {
        VStack(spacing: 10) {
            Toggle(isOn: handsFreeBinding) {
                Label("Hands-free", systemImage: "phone")
            }
            .disabled(!canTalk)
            .padding(.bottom, 2)

            Button {
                if let info = chat.info { voice.talk(info) }
            } label: {
                ZStack {
                    Circle().fill(buttonColor)
                    if voice.phase == .starting {
                        ProgressView().tint(.white)
                    } else {
                        Image(systemName: buttonIcon).font(.system(size: 32, weight: .semibold)).foregroundStyle(.white)
                    }
                }
                .frame(width: 84, height: 84)
            }
            .disabled(!canTalk)
            .accessibilityLabel(buttonLabel)

            Text(caption)
                .font(.footnote)
                .foregroundStyle(captionIsError ? Color.orange : Color.secondary)
                .multilineTextAlignment(.center)
                .frame(minHeight: 36, alignment: .top)
        }
        .padding(.horizontal, 24)
        .padding(.top, 14)
        .padding(.bottom, 8)
        .frame(maxWidth: .infinity)
        .background(.bar)
    }

    private var handsFreeBinding: Binding<Bool> {
        Binding(get: { voice.handsFree }, set: { on in
            if let info = chat.info { voice.setHandsFree(on, info: info) }
        })
    }

    private var canTalk: Bool {
        guard chat.info?.talks == true else { return false }
        if case .loading = chat.status { return false }
        return !chat.isGenerating && !chat.speech.isSpeaking
    }

    private var buttonIcon: String {
        if voice.handsFree {
            switch voice.callState {
            case .listening: return "ear"
            case .hearing: return "waveform"
            case .thinking: return "ellipsis"
            case .speaking: return "stop.fill"
            }
        }
        switch voice.phase {
        case .listening: return "arrow.up"
        case .replying: return "stop.fill"
        default: return "mic.fill"
        }
    }

    private var buttonColor: Color {
        guard canTalk else { return Color(.systemGray3) }
        if voice.handsFree { return voice.callState == .hearing ? .red : .accentColor }
        return voice.phase == .listening ? .red : .accentColor
    }

    private var buttonLabel: String {
        if voice.handsFree {
            switch voice.callState {
            case .listening: return "Listening"
            case .hearing: return "Hearing you"
            case .thinking, .speaking: return "Stop"
            }
        }
        switch voice.phase {
        case .listening: return "Send"
        case .replying: return "Stop"
        default: return "Talk"
        }
    }

    private var caption: String {
        if chat.info?.talks != true { return "Load LFM3-Audio with its projector to talk." }
        if voice.handsFree, voice.phase != .starting {
            if case .failed(let message) = voice.phase { return message }
            switch voice.callState {
            case .listening: return "Listening…"
            case .hearing: return "Hearing you…"
            case .thinking: return "Thinking…"
            case .speaking: return "Speaking · talk to interrupt"
            }
        }
        switch voice.phase {
        case .idle: return voice.exchanges.isEmpty ? "Tap to talk" : "Tap to answer"
        case .starting: return "Starting the microphone…"
        case .listening: return "Listening… tap to send"
        case .replying: return "Answering… tap to stop"
        case .failed(let message): return message
        }
    }

    private var captionIsError: Bool {
        if case .failed = voice.phase { return true }
        return false
    }
}

private struct ExchangeRow: View {
    let exchange: VoiceViewModel.Exchange
    let isReplying: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Spacer(minLength: 48)
                Group {
                    if let heard = exchange.heard, !heard.isEmpty {
                        Text(heard)
                    } else {
                        Label(String(format: "%.1f s of speech", exchange.heardSeconds), systemImage: "waveform")
                    }
                }
                .textSelection(.enabled)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .foregroundStyle(.white)
                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }

            VStack(alignment: .leading, spacing: 6) {
                if !exchange.reply.isEmpty {
                    Text(exchange.reply)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else if isReplying {
                    ProgressView().controlSize(.small)
                }
                if let error = exchange.error {
                    Label(error, systemImage: "exclamationmark.triangle").font(.footnote).foregroundStyle(.orange)
                }
                if let footer {
                    Text(footer).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    /// First sound, speed and whether playback kept up, once the reply is done.
    private var footer: String? {
        guard let s = exchange.stats else { return nil }
        var parts: [String] = []
        if let first = exchange.firstSound { parts.append(String(format: "first sound %.1f s", first)) }
        parts.append(String(format: "%.1f s of speech at %.2f× real time", s.audioSeconds, s.realTimeFactor))
        parts.append(exchange.gaps == 0 ? "no gaps" : String(format: "%d gaps (%.1f s)", exchange.gaps, exchange.gapSeconds))
        if exchange.interrupted {
            parts.append(String(format: "interrupted after %.1f s", exchange.playedSeconds ?? 0))
        } else if s.stopped {
            parts.append("stopped")
        }
        if s.hitLimit { parts.append("cut at the length limit") }
        return parts.joined(separator: " · ")
    }
}

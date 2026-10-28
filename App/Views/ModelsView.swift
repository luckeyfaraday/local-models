import SwiftUI

struct ModelsView: View {
    @Environment(ModelStore.self) private var store
    @Environment(ChatViewModel.self) private var chat
    @Environment(\.dismiss) private var dismiss
    @State private var repo = ModelStore.defaultRepo
    @State private var listing = ModelStore.Listing()
    @State private var fetching = false
    @State private var fetchError: String?

    var body: some View {
        NavigationStack {
            List {
                onDevice
                if !store.downloads.isEmpty { downloading }
                huggingFace
                memoryFooter
            }
            .navigationTitle("Models")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .task { if listing.models.isEmpty { await fetch() } }
            .refreshable {
                store.refresh()
                await fetch()
            }
            .onAppear { store.refresh() }
        }
    }

    private var onDevice: some View {
        Section("On this iPhone") {
            if store.models.isEmpty {
                Text("Nothing downloaded yet.").foregroundStyle(.secondary)
            }
            ForEach(store.models) { model in
                Button {
                    Task {
                        dismiss()
                        await chat.load(model)
                    }
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.displayName).foregroundStyle(.primary)
                            if let projector = store.projector(for: model) {
                                Text("\(Format.bytes(model.bytes)) + \(Format.bytes(projector.bytes)) projector")
                                    .font(.caption).foregroundStyle(.secondary)
                            } else {
                                Text(Format.bytes(model.bytes)).font(.caption).foregroundStyle(.secondary)
                            }
                            if model.bytes > store.availableMemory {
                                Text("Bigger than the memory iOS gives this app right now (\(Format.bytes(store.availableMemory))).")
                                    .font(.caption).foregroundStyle(.orange)
                            }
                        }
                        Spacer()
                        if chat.loadedFile == model.fileName {
                            Image(systemName: "checkmark").foregroundStyle(.tint)
                        }
                    }
                }
                .disabled(chat.isBusy)
            }
            .onDelete { offsets in
                for index in offsets {
                    let model = store.models[index]
                    Task {
                        await chat.unloadIfDeleted(model)
                        store.delete(model)
                    }
                }
            }
        }
    }

    private var downloading: some View {
        Section("Downloading") {
            ForEach(store.downloads.keys.sorted(), id: \.self) { name in
                let state = store.downloads[name] ?? .init()
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(name).font(.subheadline).lineLimit(1)
                        ProgressView(value: state.fraction)
                        Text("\(Format.bytes(state.received)) of \(state.expected > 0 ? Format.bytes(state.expected) : "?")")
                            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Button { store.cancelDownload(name) } label: {
                        Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel("Cancel download")
                }
            }
        }
    }

    private var huggingFace: some View {
        Section {
            HStack {
                TextField("owner/repo", text: $repo)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .onSubmit { Task { await fetch() } }
                if fetching {
                    ProgressView()
                } else {
                    Button("Search") { Task { await fetch() } }.buttonStyle(.borderless)
                }
            }
            if let error = fetchError ?? store.lastError {
                Text(error).font(.footnote).foregroundStyle(.orange)
            }
            ForEach(listing.models) { file in
                let projector = listing.bestProjector
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text((file.fileName as NSString).deletingPathExtension).lineLimit(1)
                            if isRecommended(file) {
                                Text("Best for phones")
                                    .font(.caption2.weight(.semibold))
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(Color.accentColor.opacity(0.15), in: Capsule())
                                    .foregroundStyle(.tint)
                            }
                        }
                        if let projector {
                            Text("\(Format.bytes(file.bytes)) + \(Format.bytes(projector.bytes)) projector")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text(Format.bytes(file.bytes)).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    let files = [file] + [projector].compactMap { $0 }
                    if files.allSatisfy({ store.isDownloaded($0) }) {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                            .accessibilityLabel("Downloaded")
                    } else if store.downloads[file.fileName] != nil
                                || (store.isDownloaded(file) && files.contains(where: { store.downloads[$0.fileName] != nil })) {
                        ProgressView()
                    } else {
                        Button { store.download(file, projector: projector) } label: {
                            Image(systemName: "arrow.down.circle").font(.title3)
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Download \(file.fileName)")
                    }
                }
            }
        } header: {
            Text("Get from Hugging Face")
        } footer: {
            Text("Any public repo with .gguf files. Vision and audio models also get their projector (mmproj). You can copy GGUF files into Files › On My iPhone › Local Models too.")
        }
    }

    private var memoryFooter: some View {
        Section {
            LabeledContent("Memory available to the app", value: Format.bytes(store.availableMemory))
            LabeledContent("Free storage", value: Format.bytes(store.freeDiskSpace))
        }
        .font(.footnote)
    }

    private func isRecommended(_ file: ModelStore.RemoteFile) -> Bool {
        // Liquid recommends Q4_0 on phones: llama.cpp repacks it into fast Arm kernels.
        file.fileName.uppercased().contains("Q4_0")
    }

    private func fetch() async {
        fetching = true
        fetchError = nil
        defer { fetching = false }
        do {
            listing = try await store.listGGUF(repo: repo)
            if listing.models.isEmpty { fetchError = "No .gguf files in \(ModelStore.normalizeRepo(repo))." }
        } catch {
            listing = ModelStore.Listing()
            fetchError = error.localizedDescription
        }
    }
}

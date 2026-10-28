import Foundation
import Observation
import os

/// GGUF files on the phone, plus downloads from Hugging Face.
/// Models live in the app's Documents folder, which the Files app shows under
/// On My iPhone › Local Models, so GGUFs can also be copied in by hand.
@Observable @MainActor
final class ModelStore {
    static let shared = ModelStore()

    struct LocalModel: Identifiable, Hashable {
        let fileName: String
        let url: URL
        let bytes: Int64
        var id: String { fileName }
        var displayName: String { (fileName as NSString).deletingPathExtension }
    }

    struct RemoteFile: Identifiable, Hashable {
        let repo: String
        let path: String
        let bytes: Int64
        var id: String { repo + "/" + path }
        var fileName: String { (path as NSString).lastPathComponent }
        var downloadURL: URL {
            let encoded = path.split(separator: "/")
                .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
                .joined(separator: "/")
            return URL(string: "https://huggingface.co/\(repo)/resolve/main/\(encoded)?download=true")!
        }
    }

    struct DownloadState: Equatable {
        var received: Int64 = 0
        var expected: Int64 = 0
        var fraction: Double { expected > 0 ? Double(received) / Double(expected) : 0 }
    }

    nonisolated static let defaultRepo = "LiquidAI/LFM3-1.7B-GGUF"

    private(set) var models: [LocalModel] = []
    private(set) var downloads: [String: DownloadState] = [:]
    var lastError: String?

    @ObservationIgnored let directory: URL
    @ObservationIgnored private var downloader: Downloader!
    @ObservationIgnored var backgroundCompletionHandler: (() -> Void)?

    private init() {
        directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        downloader = Downloader(directory: directory, store: self)
        refresh()
    }

    // MARK: - Local files

    func refresh() {
        let keys: [URLResourceKey] = [.fileSizeKey]
        let urls = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: keys)) ?? []
        models = urls
            .filter { $0.pathExtension.lowercased() == "gguf" }
            .map { url in
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                return LocalModel(fileName: url.lastPathComponent, url: url, bytes: Int64(size))
            }
            .sorted { $0.fileName.localizedStandardCompare($1.fileName) == .orderedAscending }
    }

    func model(named fileName: String) -> LocalModel? {
        models.first { $0.fileName == fileName }
    }

    func delete(_ model: LocalModel) {
        try? FileManager.default.removeItem(at: model.url)
        refresh()
    }

    func isDownloaded(_ file: RemoteFile) -> Bool {
        models.contains { $0.fileName == file.fileName }
    }

    /// Memory iOS will still let this app allocate.
    var availableMemory: Int64 {
        Int64(os_proc_available_memory())
    }

    var freeDiskSpace: Int64 {
        let values = try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    // MARK: - Hugging Face

    func listGGUF(repo input: String) async throws -> [RemoteFile] {
        let repo = Self.normalizeRepo(input)
        guard repo.split(separator: "/").count == 2,
              let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main?recursive=true") else {
            throw StoreError.badRepo(input)
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else { throw StoreError.badRepo(input) }
        guard http.statusCode == 200 else {
            throw http.statusCode == 401 || http.statusCode == 404 ? StoreError.repoNotFound(repo) : StoreError.http(http.statusCode)
        }

        struct Entry: Decodable {
            struct LFS: Decodable { let size: Int64 }
            let type: String
            let path: String
            let size: Int64?
            let lfs: LFS?
        }
        return try JSONDecoder().decode([Entry].self, from: data)
            .filter { entry in
                let lower = entry.path.lowercased()
                return entry.type == "file" && lower.hasSuffix(".gguf") && !lower.contains("mmproj")
            }
            .map { RemoteFile(repo: repo, path: $0.path, bytes: $0.lfs?.size ?? $0.size ?? 0) }
            .sorted { $0.bytes < $1.bytes }
    }

    static func normalizeRepo(_ input: String) -> String {
        var s = input.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/"] where s.hasPrefix(prefix) {
            s.removeFirst(prefix.count)
        }
        let parts = s.split(separator: "/")
        return parts.count >= 2 ? "\(parts[0])/\(parts[1])" : s
    }

    func download(_ file: RemoteFile) {
        lastError = nil
        guard downloads[file.fileName] == nil else { return }
        let needed = file.bytes + 200_000_000
        if file.bytes > 0 && freeDiskSpace < needed {
            lastError = "Not enough free space for \(file.fileName): it needs \(Format.bytes(needed)), the iPhone has \(Format.bytes(freeDiskSpace)) free."
            return
        }
        downloads[file.fileName] = DownloadState(received: 0, expected: file.bytes)
        downloader.start(url: file.downloadURL, fileName: file.fileName)
    }

    func cancelDownload(_ fileName: String) {
        downloader.cancel(fileName: fileName)
        downloads[fileName] = nil
    }

    // MARK: - Downloader callbacks

    fileprivate func downloadProgress(_ fileName: String, received: Int64, expected: Int64) {
        downloads[fileName] = DownloadState(received: received, expected: expected)
    }

    fileprivate func downloadFinished(_ fileName: String, error: String?) {
        downloads[fileName] = nil
        if let error { lastError = error }
        refresh()
    }

    fileprivate func backgroundEventsFinished() {
        backgroundCompletionHandler?()
        backgroundCompletionHandler = nil
    }

    enum StoreError: LocalizedError {
        case badRepo(String)
        case repoNotFound(String)
        case http(Int)

        var errorDescription: String? {
            switch self {
            case .badRepo(let s): return "\"\(s)\" isn't a Hugging Face repo. Use owner/name, like \(ModelStore.defaultRepo)."
            case .repoNotFound(let s): return "\(s) doesn't exist or is private."
            case .http(let code): return "Hugging Face answered HTTP \(code)."
            }
        }
    }
}

/// Background URLSession, so a 1 GB download keeps going when the phone locks.
private final class Downloader: NSObject, URLSessionDownloadDelegate {
    private let directory: URL
    private weak var store: ModelStore?
    private var session: URLSession!
    private var lastReport: [String: Int64] = [:]
    private let lock = NSLock()

    init(directory: URL, store: ModelStore) {
        self.directory = directory
        self.store = store
        super.init()
        let config = URLSessionConfiguration.background(withIdentifier: "io.github.luckeyfaraday.localmodels.downloads")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        // Pick up downloads that survived an app relaunch.
        session.getAllTasks { tasks in
            for task in tasks {
                guard let name = task.taskDescription, task.state == .running || task.state == .suspended else { continue }
                let received = task.countOfBytesReceived
                let expected = task.countOfBytesExpectedToReceive
                Task { @MainActor in self.store?.downloadProgress(name, received: received, expected: expected) }
            }
        }
    }

    func start(url: URL, fileName: String) {
        let task = session.downloadTask(with: url)
        task.taskDescription = fileName
        task.resume()
    }

    func cancel(fileName: String) {
        session.getAllTasks { tasks in
            tasks.filter { $0.taskDescription == fileName }.forEach { $0.cancel() }
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let name = downloadTask.taskDescription else { return }
        lock.lock()
        let last = lastReport[name] ?? 0
        let report = totalBytesWritten - last >= 4_000_000
        if report { lastReport[name] = totalBytesWritten }
        lock.unlock()
        guard report else { return }
        Task { @MainActor in
            self.store?.downloadProgress(name, received: totalBytesWritten, expected: totalBytesExpectedToWrite)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let name = downloadTask.taskDescription else { return }
        lock.lock()
        lastReport[name] = nil
        lock.unlock()

        var failure: String?
        if let http = downloadTask.response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            failure = "Downloading \(name) failed: HTTP \(http.statusCode)."
        } else {
            // The file has to be moved before this method returns.
            var destination = directory.appendingPathComponent(name)
            do {
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: location, to: destination)
                var values = URLResourceValues()
                values.isExcludedFromBackup = true
                try? destination.setResourceValues(values)
            } catch {
                failure = "Couldn't save \(name): \(error.localizedDescription)"
            }
        }
        Task { @MainActor in self.store?.downloadFinished(name, error: failure) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error, let name = task.taskDescription else { return }
        let cancelled = (error as NSError).code == NSURLErrorCancelled
        let message = cancelled ? nil : "Downloading \(name) failed: \(error.localizedDescription)"
        Task { @MainActor in self.store?.downloadFinished(name, error: message) }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in self.store?.backgroundEventsFinished() }
    }
}

enum Format {
    static func bytes(_ n: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: n, countStyle: .file)
    }

    static func params(_ n: UInt64) -> String {
        let b = Double(n) / 1e9
        return b >= 1 ? String(format: "%.2fB params", b) : String(format: "%.0fM params", Double(n) / 1e6)
    }
}

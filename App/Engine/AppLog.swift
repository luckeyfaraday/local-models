import Foundation
import llama
import MetricKit
import os
import UIKit

/// A log the phone keeps by itself, since there's no Mac attached to read the
/// console: what llama.cpp printed (it goes to an invisible stderr otherwise)
/// and the app's own notes. It lives in Files › On My iPhone › Galerna Local
/// Models › Logs, next to the crash reports iOS hands over (see Diagnostics).
/// `log.txt` is this run, `log-previous.txt` the one before: after a crash or
/// a kill for memory, that's where its last lines are.
enum AppLog {
    static let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Logs", isDirectory: true)
    static let current = folder.appendingPathComponent("log.txt")
    static let previous = folder.appendingPathComponent("log-previous.txt")
    /// One run writes at most this much.
    static let maxBytes = 2_000_000

    static var commit: String { info("GitCommit") }
    static var version: String { info("CFBundleShortVersionString") }
    static var llamaBuild: String { info("LlamaBuild") }

    private static let lock = NSLock()
    private static var handle: FileHandle?
    private static var written = 0
    /// llama.cpp sends a line in pieces: the pieces after a skipped one are skipped too.
    private static var skipping = false
    private static let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()

    /// Call once at launch, before a model is loaded.
    @MainActor static func start() {
        let files = FileManager.default
        try? files.createDirectory(at: folder, withIntermediateDirectories: true)
        try? files.removeItem(at: previous)
        try? files.moveItem(at: current, to: previous)
        files.createFile(atPath: current.path, contents: nil)
        lock.lock()
        handle = try? FileHandle(forWritingTo: current)
        lock.unlock()

        var system = utsname()
        uname(&system)
        let machine = withUnsafeBytes(of: &system.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        note("Galerna Local Models \(version) (\(commit)), llama.cpp \(llamaBuild), \(machine), iOS \(UIDevice.current.systemVersion)")

        llama_log_set({ level, text, _ in AppLog.engine(level, text) }, nil)
        mtmd_helper_log_set({ level, text, _ in AppLog.engine(level, text) }, nil)

        NotificationCenter.default.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil,
                                               queue: nil) { _ in
            AppLog.note("memory warning: \(os_proc_available_memory() / 1_048_576) MB left")
        }
    }

    /// One line of the app's own, with the time.
    static func note(_ line: String) {
        lock.lock()
        defer { lock.unlock() }
        append(Data("\(clock.string(from: Date())) \(line)\n".utf8))
    }

    private static func engine(_ level: ggml_log_level, _ text: UnsafePointer<CChar>?) {
        guard let text else { return }
        lock.lock()
        defer { lock.unlock() }
        if level != GGML_LOG_LEVEL_CONT { skipping = level == GGML_LOG_LEVEL_DEBUG }
        if skipping { return }
        append(Data(bytes: text, count: strlen(text)))
    }

    /// Under the lock. Written straight through, so the last line before a crash is on disk.
    private static func append(_ data: Data) {
        guard let handle, written < maxBytes else { return }
        written += data.count
        try? handle.write(contentsOf: data)
        if written >= maxBytes { try? handle.write(contentsOf: Data("\n(log full)\n".utf8)) }
    }

    private static func info(_ key: String) -> String {
        let value = Bundle.main.object(forInfoDictionaryKey: key) as? String ?? ""
        return value.isEmpty ? "?" : value
    }
}

/// Crash, hang and exit reports from iOS (MetricKit). A crash report arrives
/// at the next launch and is saved as JSON in the Logs folder; once a day iOS
/// also says how the app's runs ended, which is the only place a kill for
/// memory shows up.
final class Diagnostics: NSObject, MXMetricManagerSubscriber {
    static let shared = Diagnostics()
    private static let kept = 20

    func start() {
        MXMetricManager.shared.add(self)
    }

    func didReceive(_ payloads: [MXDiagnosticPayload]) {
        for payload in payloads {
            let name = "diagnostics-\(Int(payload.timeStampEnd.timeIntervalSince1970)).json"
            try? payload.jsonRepresentation().write(to: AppLog.folder.appendingPathComponent(name), options: .atomic)
            AppLog.note("report from iOS saved as \(name): \(payload.crashDiagnostics?.count ?? 0) crashes, "
                + "\(payload.hangDiagnostics?.count ?? 0) hangs, \(payload.cpuExceptionDiagnostics?.count ?? 0) CPU exceptions")
        }
        let files = FileManager.default
        let reports = ((try? files.contentsOfDirectory(at: AppLog.folder, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("diagnostics-") }
            .sorted { $0.lastPathComponent.count == $1.lastPathComponent.count
                ? $0.lastPathComponent < $1.lastPathComponent : $0.lastPathComponent.count < $1.lastPathComponent.count }
        for old in reports.dropLast(Self.kept) { try? files.removeItem(at: old) }
    }

    func didReceive(_ payloads: [MXMetricPayload]) {
        for payload in payloads {
            guard let exits = payload.applicationExitMetrics else { continue }
            let front = exits.foregroundExitData, back = exits.backgroundExitData
            AppLog.note("exits in the day to \(payload.timeStampEnd): on screen \(front.cumulativeNormalAppExitCount) normal, "
                + "\(front.cumulativeMemoryResourceLimitExitCount) over the memory limit, "
                + "\(front.cumulativeBadAccessExitCount) bad access, \(front.cumulativeAbnormalExitCount) abnormal, "
                + "\(front.cumulativeAppWatchdogExitCount) watchdog; in the background "
                + "\(back.cumulativeMemoryPressureExitCount) for memory pressure, "
                + "\(back.cumulativeMemoryResourceLimitExitCount) over the memory limit, "
                + "\(back.cumulativeAbnormalExitCount) abnormal")
        }
    }
}

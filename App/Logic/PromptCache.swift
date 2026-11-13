import Foundation

/// The model's memory after a chat turn holds the prompt and the reply. A
/// hybrid model (LFM) can't drop part of that, but it can take more: when the
/// next prompt is the same text with more after it, only the new part is read.
enum PromptCache {
    /// Where the new part of `prompt` starts, if `prompt` is `cached` with
    /// something added. Both are UTF-8. Nil if it differs or adds nothing.
    static func newPartStart(of prompt: [UInt8], after cached: [UInt8]) -> Int? {
        guard !cached.isEmpty, prompt.count > cached.count, prompt.starts(with: cached) else { return nil }
        return cached.count
    }
}

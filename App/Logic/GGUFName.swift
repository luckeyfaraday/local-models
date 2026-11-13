import Foundation

/// What a GGUF's file name says about it.
enum GGUFName {
    /// "Q4_0" for "LFM3-Audio-2B-Q4_0.gguf": the quantization at the end of a file name.
    static func quantTag(_ fileName: String) -> String? {
        let stem = (fileName as NSString).deletingPathExtension
        guard let range = stem.range(of: #"[-_.](i?q[0-9][a-z0-9_]*|bf16|f16|f32)$"#,
                                     options: [.regularExpression, .caseInsensitive]) else { return nil }
        return String(stem[range].dropFirst()).uppercased()
    }

    /// "lfm3-audio-2b" for "mmproj-LFM3-Audio-2B-Q8_0.gguf".
    static func projectorBase(_ fileName: String) -> String {
        var stem = (fileName as NSString).deletingPathExtension.lowercased()
        for marker in ["mmproj-", "mmproj_", "-mmproj", "_mmproj", ".mmproj", "mmproj"] {
            stem = stem.replacingOccurrences(of: marker, with: "")
        }
        if let tag = quantTag(stem + ".gguf") {
            stem = String(stem.dropLast(tag.count + 1))
        }
        return stem
    }
}

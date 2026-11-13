import UIKit

/// An image attached to a chat message: an upright JPEG for the model and a
/// small copy to show in the chat.
struct ChatImage: Identifiable, Equatable, @unchecked Sendable {
    let id = UUID()
    let jpeg: Data
    let thumbnail: UIImage

    /// The model sees at most `LlamaEngine.imageMaxTokens` × 1024 pixels
    /// (about 1 MP), so bigger photos are scaled down before they're kept.
    static let maxSide: CGFloat = 1536
    static let thumbnailSide: CGFloat = 640

    static func == (a: ChatImage, b: ChatImage) -> Bool { a.id == b.id }

    /// Applies the photo's orientation, flattens transparency onto white and
    /// scales it down. Runs off the main thread: a 12 MP photo takes a moment.
    static func make(from data: Data) async -> ChatImage? {
        await Task.detached(priority: .userInitiated) {
            UIImage(data: data).flatMap(ChatImage.init)
        }.value
    }

    static func make(from image: UIImage) async -> ChatImage? {
        await Task.detached(priority: .userInitiated) { ChatImage(image) }.value
    }

    private init?(_ image: UIImage) {
        let pixels = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
        guard pixels.width >= 1, pixels.height >= 1 else { return nil }
        let upright = Self.render(image, fitting: Self.maxSide, pixels: pixels)
        guard let jpeg = upright.jpegData(compressionQuality: 0.9) else { return nil }
        self.jpeg = jpeg
        thumbnail = Self.render(upright, fitting: Self.thumbnailSide, pixels: upright.size)
    }

    private static func render(_ image: UIImage, fitting side: CGFloat, pixels: CGSize) -> UIImage {
        let scale = min(1, side / max(pixels.width, pixels.height))
        let size = CGSize(width: max(1, (pixels.width * scale).rounded()), height: max(1, (pixels.height * scale).rounded()))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        format.preferredRange = .standard
        return UIGraphicsImageRenderer(size: size, format: format).image { context in
            UIColor.white.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            image.draw(in: CGRect(origin: .zero, size: size))
        }
    }
}

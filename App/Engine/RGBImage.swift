import CoreGraphics
import Foundation
import ImageIO

/// Packed 8-bit RGB pixels, the layout `mtmd_bitmap_init` takes.
struct RGBImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    /// Decodes JPEG, PNG or HEIC data. EXIF orientation isn't applied: the chat
    /// stores its images upright already. Transparency is flattened onto white.
    init?(data: Data) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let w = image.width, h = image.height
        guard w > 0, h > 0 else { return nil }

        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                          bytesPerRow: w * 4, space: space,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            let rect = CGRect(x: 0, y: 0, width: w, height: h)
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(rect)
            context.draw(image, in: rect)
            return true
        }
        guard drawn else { return nil }

        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        rgb.withUnsafeMutableBufferPointer { out in
            rgba.withUnsafeBufferPointer { src in
                for i in 0..<(w * h) {
                    out[i * 3] = src[i * 4]
                    out[i * 3 + 1] = src[i * 4 + 1]
                    out[i * 3 + 2] = src[i * 4 + 2]
                }
            }
        }
        width = w
        height = h
        pixels = rgb
    }
}

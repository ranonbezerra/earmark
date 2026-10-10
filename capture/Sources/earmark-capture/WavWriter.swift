import Foundation

/// Minimal 16-bit mono PCM WAV writer.
///
/// The header is rewritten every few seconds, so a file left behind by a crash
/// or a hard kill is still a valid WAV up to the last refresh.
final class WavWriter {
    private static let headerRefreshSeconds = 5

    let url: URL
    private let handle: FileHandle
    private let sampleRate: Int
    private(set) var frames = 0
    private var framesAtLastRefresh = 0

    init(url: URL, sampleRate: Int) throws {
        self.url = url
        self.sampleRate = sampleRate
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw CaptureError("cannot create \(url.path)")
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: header())
    }

    func write(_ samples: UnsafePointer<Int16>, count: Int) throws {
        guard count > 0 else { return }
        let data = Data(bytes: samples, count: count * MemoryLayout<Int16>.size)
        try handle.write(contentsOf: data)
        frames += count
        if frames - framesAtLastRefresh >= Self.headerRefreshSeconds * sampleRate {
            try refreshHeader()
        }
    }

    func close() {
        try? refreshHeader()
        try? handle.close()
    }

    private func refreshHeader() throws {
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: header())
        try handle.seekToEnd()
        framesAtLastRefresh = frames
    }

    private func header() -> Data {
        let bytesPerSample = 2
        let dataBytes = UInt32(frames * bytesPerSample)
        var data = Data(capacity: 44)
        data.append(contentsOf: Array("RIFF".utf8))
        data.appendLittleEndian(UInt32(36) + dataBytes)
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        data.appendLittleEndian(UInt32(16))                               // fmt chunk size
        data.appendLittleEndian(UInt16(1))                                // PCM
        data.appendLittleEndian(UInt16(1))                                // mono
        data.appendLittleEndian(UInt32(sampleRate))
        data.appendLittleEndian(UInt32(sampleRate * bytesPerSample))      // byte rate
        data.appendLittleEndian(UInt16(bytesPerSample))                   // block align
        data.appendLittleEndian(UInt16(16))                               // bits per sample
        data.append(contentsOf: Array("data".utf8))
        data.appendLittleEndian(dataBytes)
        return data
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}

import AVFoundation
import Foundation

/// Appends one line per segment to `segments.jsonl` as soon as the segment is
/// opened, so the timeline survives even if the process is killed.
final class Manifest {
    private let handle: FileHandle
    private let lock = NSLock()

    init(directory: URL) throws {
        let url = directory.appendingPathComponent("segments.jsonl")
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()
    }

    func append(_ entry: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]) else { return }
        lock.lock()
        defer { lock.unlock() }
        try? handle.write(contentsOf: data + Data("\n".utf8))
    }

    func close() {
        try? handle.close()
    }
}

/// Receives converted 16 kHz mono Int16 audio for one track ("mic" or
/// "system") and writes it as fixed length WAV segments.
final class TrackSink {
    let track: String
    private let directory: URL
    private let sampleRate: Int
    private let segmentFrames: Int
    private let sessionStart: Date
    private let manifest: Manifest

    private let lock = NSLock()
    private var writer: WavWriter?
    private var index = 0
    private var peak: Int32 = 0
    private var closed = false
    private(set) var totalFrames = 0

    init(track: String, directory: URL, sampleRate: Int, segmentSeconds: Int, sessionStart: Date, manifest: Manifest) {
        self.track = track
        self.directory = directory
        self.sampleRate = sampleRate
        self.segmentFrames = max(1, segmentSeconds) * sampleRate
        self.sessionStart = sessionStart
        self.manifest = manifest
    }

    func write(_ buffer: AVAudioPCMBuffer) {
        guard let samples = buffer.int16ChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }

        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }

        var localPeak = peak
        for i in 0..<count {
            let magnitude = abs(Int32(samples[i]))
            if magnitude > localPeak { localPeak = magnitude }
        }
        peak = localPeak

        // The buffer has just been delivered, so its last frame is "now".
        let bufferEnd = Date().timeIntervalSince(sessionStart)
        var offset = 0
        do {
            while offset < count {
                if writer == nil {
                    let remaining = Double(count - offset) / Double(sampleRate)
                    try openSegment(startOffset: max(0, bufferEnd - remaining))
                }
                guard let writer else { break }
                let chunk = min(segmentFrames - writer.frames, count - offset)
                try writer.write(samples + offset, count: chunk)
                offset += chunk
                totalFrames += chunk
                if writer.frames >= segmentFrames {
                    closeSegment()
                }
            }
        } catch {
            closed = true
            closeSegment()
            Output.emit("error", ["track": track, "message": "write failed: \(error)"])
        }
    }

    /// Peak level since the previous call, in dBFS (floored at -120).
    func takePeakDecibels() -> Double {
        lock.lock()
        defer { lock.unlock() }
        let value = peak
        peak = 0
        guard value > 0 else { return -120 }
        return max(-120, 20 * log10(Double(value) / 32768.0))
    }

    /// Starts a new segment on the next write. Called after a capture gap
    /// (device change), so the audio that follows gets a fresh start offset
    /// instead of being glued to the audio from before the gap.
    func rotate() {
        lock.lock()
        defer { lock.unlock() }
        closeSegment()
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        closed = true
        closeSegment()
    }

    private func openSegment(startOffset: Double) throws {
        let name = String(format: "%@-%05d.wav", track, index)
        writer = try WavWriter(url: directory.appendingPathComponent(name), sampleRate: sampleRate)
        manifest.append([
            "track": track,
            "index": index,
            "file": name,
            "start_offset": (startOffset * 1000).rounded() / 1000,
        ])
    }

    private func closeSegment() {
        guard let current = writer else { return }
        current.close()
        Output.emit("segment", [
            "track": track,
            "index": index,
            "file": current.url.lastPathComponent,
            "frames": current.frames,
        ])
        writer = nil
        index += 1
    }
}

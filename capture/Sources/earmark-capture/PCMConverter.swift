import AVFoundation

/// Streams audio from whatever format a device delivers into 16 kHz mono
/// Int16, which is what Whisper models expect.
final class PCMConverter {
    static let outputSampleRate = 16_000

    let outputFormat: AVAudioFormat
    private let inputFormat: AVAudioFormat
    private let converter: AVAudioConverter

    init(from inputFormat: AVAudioFormat) throws {
        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: Double(Self.outputSampleRate),
            channels: 1,
            interleaved: true
        ) else {
            throw CaptureError("cannot create the 16 kHz output format")
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw CaptureError("cannot convert from \(inputFormat)")
        }
        if inputFormat.channelCount == 2 {
            // Mix left and right instead of keeping only the first channel.
            converter.downmix = true
        }
        self.inputFormat = inputFormat
        self.outputFormat = outputFormat
        self.converter = converter
    }

    /// Converts one buffer. The converter keeps its resampling state between
    /// calls, so consecutive buffers join without clicks.
    func convert(_ input: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard input.frameLength > 0 else { return nil }
        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let capacity = AVAudioFrameCount((Double(input.frameLength) * ratio).rounded(.up)) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return nil }

        var supplied = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .noDataNow
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return input
        }
        if status == .error {
            return nil
        }
        return output.frameLength > 0 ? output : nil
    }
}

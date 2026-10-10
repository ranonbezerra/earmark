import AVFoundation
import Foundation

/// Records the default input device through AVAudioEngine.
final class MicCapture {
    private let engine = AVAudioEngine()
    private let sink: TrackSink
    private let lock = NSLock()
    private var converter: PCMConverter?
    private var observer: NSObjectProtocol?
    private var stopped = false

    init(sink: TrackSink) {
        self.sink = sink
    }

    /// Asks for microphone access if it has not been decided yet.
    static func requestAccess() -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            let semaphore = DispatchSemaphore(value: 0)
            var granted = false
            AVCaptureDevice.requestAccess(for: .audio) { allowed in
                granted = allowed
                semaphore.signal()
            }
            semaphore.wait()
            return granted
        default:
            return false
        }
    }

    static var authorizationDescription: String {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return "authorized"
        case .notDetermined: return "not_determined"
        case .denied: return "denied"
        case .restricted: return "restricted"
        @unknown default: return "unknown"
        }
    }

    func start() throws {
        try startEngine()
        // Fired when the input device changes (for example AirPods connecting).
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: nil
        ) { [weak self] _ in
            // Restart on the main queue (the same one stop() uses) and give
            // the new route a moment to settle before reading its format.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                self?.restart()
            }
        }
    }

    func stop() {
        stopped = true
        if let observer {
            NotificationCenter.default.removeObserver(observer)
        }
        observer = nil
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }

    private func startEngine() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw CaptureError("no audio input device available")
        }
        let newConverter = try PCMConverter(from: format)
        lock.lock()
        converter = newConverter
        lock.unlock()

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            self.lock.lock()
            let current = self.converter
            self.lock.unlock()
            if let converted = current?.convert(buffer) {
                self.sink.write(converted)
            }
        }
        engine.prepare()
        try engine.start()
    }

    private func restart() {
        guard !stopped else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        sink.rotate()
        do {
            try startEngine()
            Output.emit("device_changed", ["track": "mic"])
        } catch {
            Output.emit("error", ["track": "mic", "message": "restart failed: \(error)"])
        }
    }
}

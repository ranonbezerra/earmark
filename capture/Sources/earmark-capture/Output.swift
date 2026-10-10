import Foundation

/// Line-delimited JSON events on stdout. The Python core reads these to know
/// when recording is live, which tracks are active and when it stopped.
enum Output {
    private static let lock = NSLock()

    static func emit(_ event: String, _ fields: [String: Any] = [:]) {
        var payload = fields
        payload["event"] = event
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
              let line = String(data: data, encoding: .utf8)
        else {
            log("could not encode event \(event)")
            return
        }
        lock.lock()
        defer { lock.unlock() }
        print(line)
        fflush(stdout)
    }

    /// Human readable diagnostics go to stderr so stdout stays machine readable.
    static func log(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}

struct CaptureError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}

import AVFoundation
import Foundation

let version = "0.1.0"

let usage = """
earmark-capture \(version)

Records the microphone and the system audio output into 16 kHz mono WAV
segments (mic-00000.wav, system-00000.wav, ...) plus a segments.jsonl manifest.
Progress is reported as JSON lines on stdout. Stop with Ctrl+C (SIGINT) or
SIGTERM; files are finalized on the way out.

USAGE:
  earmark-capture record --out-dir <dir> [--no-mic] [--no-system]
                         [--segment-seconds <n>] [--max-seconds <n>]
  earmark-capture check
  earmark-capture --version
"""

struct Options {
    var command = "record"
    var outDir: String?
    var mic = true
    var system = true
    var segmentSeconds = 300
    var maxSeconds: Int?
}

func fail(_ message: String) -> Never {
    Output.log("error: \(message)\n\n\(usage)")
    exit(2)
}

func parseOptions() -> Options {
    var options = Options()
    var arguments = Array(CommandLine.arguments.dropFirst())
    if let first = arguments.first, !first.hasPrefix("-") {
        options.command = first
        arguments.removeFirst()
    }

    func intValue(_ flag: String, _ raw: String?) -> Int {
        guard let raw, let value = Int(raw), value > 0 else { fail("\(flag) needs a positive number") }
        return value
    }

    var index = 0
    while index < arguments.count {
        let argument = arguments[index]
        let next: String? = index + 1 < arguments.count ? arguments[index + 1] : nil
        switch argument {
        case "--out-dir":
            guard let next else { fail("--out-dir needs a path") }
            options.outDir = next
            index += 1
        case "--segment-seconds":
            options.segmentSeconds = intValue(argument, next)
            index += 1
        case "--max-seconds":
            options.maxSeconds = intValue(argument, next)
            index += 1
        case "--no-mic":
            options.mic = false
        case "--no-system":
            options.system = false
        case "--version":
            print(version)
            exit(0)
        case "-h", "--help":
            print(usage)
            exit(0)
        default:
            fail("unknown argument \(argument)")
        }
        index += 1
    }
    return options
}

func runCheck() -> Never {
    var report: [String: Any] = [
        "version": version,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "microphone_access": MicCapture.authorizationDescription,
    ]
    if let device = try? CoreAudioHelpers.defaultInputDevice() {
        report["input_device"] = (try? CoreAudioHelpers.deviceName(device)) ?? "unknown"
    }
    if let device = try? CoreAudioHelpers.defaultOutputDevice() {
        report["output_device"] = (try? CoreAudioHelpers.deviceName(device)) ?? "unknown"
    }
    Output.emit("check", report)
    exit(0)
}

let options = parseOptions()

switch options.command {
case "check":
    runCheck()
case "record":
    break
default:
    fail("unknown command \(options.command)")
}

guard let outDir = options.outDir else { fail("record needs --out-dir") }
guard options.mic || options.system else { fail("nothing to record: both tracks are disabled") }

let directory = URL(fileURLWithPath: outDir, isDirectory: true)
let manifest: Manifest
do {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    manifest = try Manifest(directory: directory)
} catch {
    Output.emit("fatal", ["message": "cannot prepare \(directory.path): \(error)"])
    exit(1)
}

let sessionStart = Date()
let parentPID = getppid()
var sinks: [TrackSink] = []
var micCapture: MicCapture?
var systemCapture: SystemCapture?
var stopping = false

func shutdown(_ reason: String) {
    guard !stopping else { return }
    stopping = true
    // Finalize the files first: that is the part that must not be lost. If
    // Core Audio then hangs while tearing down, the watchdog still exits.
    DispatchQueue.global().asyncAfter(deadline: .now() + 3) { _exit(0) }
    var frames: [String: Any] = [:]
    for sink in sinks {
        sink.close()
        frames[sink.track] = sink.totalFrames
    }
    manifest.close()
    Output.emit("stopped", [
        "reason": reason,
        "duration": (Date().timeIntervalSince(sessionStart) * 1000).rounded() / 1000,
        "frames": frames,
    ])
    micCapture?.stop()
    systemCapture?.stop()
    exit(0)
}

// Signals are handled on the main queue instead of in a signal handler. They
// are set up before capture starts, so a Ctrl+C that arrives during the
// permission prompts is still honored once the main queue runs. A closed
// stdout (the parent went away) must not kill us before files are finalized.
signal(SIGPIPE, SIG_IGN)
signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
let signalSources: [DispatchSourceSignal] = [SIGINT, SIGTERM].map { number in
    let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
    source.setEventHandler { shutdown("signal") }
    source.resume()
    return source
}

func makeSink(_ track: String) -> TrackSink {
    TrackSink(
        track: track,
        directory: directory,
        sampleRate: PCMConverter.outputSampleRate,
        segmentSeconds: options.segmentSeconds,
        sessionStart: sessionStart,
        manifest: manifest
    )
}

// The microphone goes first: opening it can switch a Bluetooth headset to
// its call profile, which changes the output format the system tap reads.
if options.mic {
    if MicCapture.requestAccess() {
        let sink = makeSink("mic")
        let capture = MicCapture(sink: sink)
        do {
            try capture.start()
            micCapture = capture
            sinks.append(sink)
        } catch {
            Output.emit("error", ["track": "mic", "message": "\(error)"])
        }
    } else {
        Output.emit("error", [
            "track": "mic",
            "message": "microphone access is \(MicCapture.authorizationDescription); allow it in System Settings > Privacy & Security > Microphone",
        ])
    }
}

if options.system {
    let sink = makeSink("system")
    let capture = SystemCapture(sink: sink)
    do {
        try capture.start()
        systemCapture = capture
        sinks.append(sink)
    } catch {
        Output.emit("error", ["track": "system", "message": "\(error)"])
    }
}

guard !sinks.isEmpty else {
    Output.emit("fatal", ["message": "no track could be started"])
    exit(1)
}

Output.emit("ready", [
    "mic": micCapture != nil,
    "system": systemCapture != nil,
    "sample_rate": PCMConverter.outputSampleRate,
    "segment_seconds": options.segmentSeconds,
    "out_dir": directory.path,
    "started_at": ISO8601DateFormatter().string(from: sessionStart),
])

let ticker = DispatchSource.makeTimerSource(queue: .main)
ticker.schedule(deadline: .now() + 2, repeating: 2)
ticker.setEventHandler {
    // Never keep recording if whoever launched us is gone.
    if getppid() != parentPID {
        shutdown("parent_exited")
    }
    if let limit = options.maxSeconds, Date().timeIntervalSince(sessionStart) >= Double(limit) {
        shutdown("max_seconds")
    }
    // A system track that stays at -120 dB while audio is playing means the
    // system audio recording permission was not granted: macOS then delivers
    // silence instead of an error.
    var levels: [String: Any] = [:]
    for sink in sinks {
        levels[sink.track] = (sink.takePeakDecibels() * 10).rounded() / 10
    }
    Output.emit("level", ["peak_db": levels])
}
ticker.resume()

dispatchMain()

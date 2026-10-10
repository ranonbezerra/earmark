// swift-tools-version: 5.9
import PackageDescription

// Command line helper that records the microphone and the system audio output
// into 16 kHz mono WAV segments. Core Audio process taps need macOS 14.2+.
let package = Package(
    name: "ouvinte-capture",
    platforms: [.macOS("14.2")],
    targets: [
        .executableTarget(
            name: "ouvinte-capture",
            path: "Sources/ouvinte-capture",
            exclude: ["Info.plist"],
            linkerSettings: [
                // Embed the Info.plist so the binary carries its own usage
                // descriptions for the microphone and system audio prompts.
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/ouvinte-capture/Info.plist",
                ])
            ]
        )
    ]
)

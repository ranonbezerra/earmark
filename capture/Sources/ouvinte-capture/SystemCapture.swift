import AudioToolbox
import AVFoundation
import CoreAudio
import Foundation

/// Records everything the Mac plays, using a Core Audio process tap
/// (macOS 14.2+). The tap is global and unmuted: audio keeps playing normally
/// while a copy is delivered to us through a private aggregate device.
final class SystemCapture {
    private static let maxRestartAttempts = 3

    private let sink: TrackSink
    // Two queues on purpose. Core Audio delivers buffers on `ioQueue` and
    // AudioDeviceStop waits for that delivery to finish, so stopping the
    // device from the same queue could deadlock. All setup and teardown
    // happens on `control`.
    private let control = DispatchQueue(label: "ouvinte.capture.system.control")
    private let ioQueue = DispatchQueue(label: "ouvinte.capture.system.io", qos: .userInitiated)

    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var listening = false
    private var stopped = false

    private var defaultOutputAddress = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    private lazy var outputChanged: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
        self?.restart(attempt: 1)
    }

    init(sink: TrackSink) {
        self.sink = sink
    }

    func start() throws {
        try control.sync {
            try self.setUp()
            // The aggregate device is clocked by the current output device, so
            // it has to be rebuilt when the user switches outputs (headphones).
            let status = AudioObjectAddPropertyListenerBlock(
                AudioObjectID(kAudioObjectSystemObject), &self.defaultOutputAddress, self.control, self.outputChanged
            )
            self.listening = status == noErr
        }
    }

    func stop() {
        control.sync {
            self.stopped = true
            if self.listening {
                AudioObjectRemovePropertyListenerBlock(
                    AudioObjectID(kAudioObjectSystemObject), &self.defaultOutputAddress, self.control, self.outputChanged
                )
                self.listening = false
            }
            self.tearDown()
        }
    }

    // MARK: - Tap lifecycle (always on `control`)

    private func setUp() throws {
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        description.uuid = UUID()
        description.name = "ouvinte"
        description.isPrivate = true
        description.muteBehavior = .unmuted

        var newTap = AudioObjectID(kAudioObjectUnknown)
        try check(AudioHardwareCreateProcessTap(description, &newTap), "create process tap")
        tapID = newTap

        do {
            var streamDescription = try CoreAudioHelpers.tapFormat(tapID)
            guard let tapFormat = AVAudioFormat(streamDescription: &streamDescription) else {
                throw CaptureError("unsupported tap format")
            }
            let converter = try PCMConverter(from: tapFormat)

            let outputUID = try CoreAudioHelpers.deviceUID(CoreAudioHelpers.defaultOutputDevice())
            let aggregate: [String: Any] = [
                kAudioAggregateDeviceNameKey: "ouvinte-capture",
                kAudioAggregateDeviceUIDKey: UUID().uuidString,
                kAudioAggregateDeviceMainSubDeviceKey: outputUID,
                kAudioAggregateDeviceIsPrivateKey: true,
                kAudioAggregateDeviceIsStackedKey: false,
                kAudioAggregateDeviceTapAutoStartKey: true,
                kAudioAggregateDeviceSubDeviceListKey: [
                    [kAudioSubDeviceUIDKey: outputUID],
                ],
                kAudioAggregateDeviceTapListKey: [
                    [
                        kAudioSubTapDriftCompensationKey: true,
                        kAudioSubTapUIDKey: description.uuid.uuidString,
                    ],
                ],
            ]
            var newAggregate = AudioObjectID(kAudioObjectUnknown)
            try check(
                AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &newAggregate),
                "create aggregate device"
            )
            aggregateID = newAggregate

            // The block owns everything it needs, so it never reads state
            // that teardown is changing on the other queue.
            let sink = self.sink
            var newProc: AudioDeviceIOProcID?
            try check(
                AudioDeviceCreateIOProcIDWithBlock(&newProc, aggregateID, ioQueue) { _, input, _, _, _ in
                    guard let buffer = AVAudioPCMBuffer(
                        pcmFormat: tapFormat, bufferListNoCopy: input, deallocator: nil
                    ), let converted = converter.convert(buffer) else { return }
                    sink.write(converted)
                },
                "create IO proc"
            )
            procID = newProc
            try check(AudioDeviceStart(aggregateID, procID), "start aggregate device")
        } catch {
            tearDown()
            throw error
        }
    }

    private func tearDown() {
        if aggregateID != kAudioObjectUnknown {
            if let procID {
                AudioDeviceStop(aggregateID, procID)
                AudioDeviceDestroyIOProcID(aggregateID, procID)
                // Let a block that was already queued finish before the
                // device and its buffers go away.
                ioQueue.sync {}
            }
            AudioHardwareDestroyAggregateDevice(aggregateID)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
        }
        procID = nil
        aggregateID = AudioObjectID(kAudioObjectUnknown)
        tapID = AudioObjectID(kAudioObjectUnknown)
    }

    private func restart(attempt: Int) {
        guard !stopped else { return }
        tearDown()
        sink.rotate()
        do {
            try setUp()
            Output.emit("device_changed", ["track": "system"])
        } catch {
            guard attempt < Self.maxRestartAttempts else {
                Output.emit("error", ["track": "system", "message": "restart failed: \(error)"])
                return
            }
            // A device that was just plugged in may need a moment.
            control.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.restart(attempt: attempt + 1)
            }
        }
    }

    private func check(_ status: OSStatus, _ action: String) throws {
        guard status == noErr else {
            throw CaptureError("could not \(action) (OSStatus \(status))")
        }
    }
}

enum CoreAudioHelpers {
    static func defaultOutputDevice() throws -> AudioDeviceID {
        try device(for: kAudioHardwarePropertyDefaultOutputDevice, label: "output")
    }

    static func defaultInputDevice() throws -> AudioDeviceID {
        try device(for: kAudioHardwarePropertyDefaultInputDevice, label: "input")
    }

    static func deviceUID(_ device: AudioDeviceID) throws -> String {
        try stringProperty(device, kAudioDevicePropertyDeviceUID)
    }

    static func deviceName(_ device: AudioDeviceID) throws -> String {
        try stringProperty(device, kAudioObjectPropertyName)
    }

    static func tapFormat(_ tap: AudioObjectID) throws -> AudioStreamBasicDescription {
        var address = globalAddress(kAudioTapPropertyFormat)
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var description = AudioStreamBasicDescription()
        let status = AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &description)
        guard status == noErr else {
            throw CaptureError("could not read the tap format (OSStatus \(status))")
        }
        return description
    }

    private static func device(for selector: AudioObjectPropertySelector, label: String) throws -> AudioDeviceID {
        var address = globalAddress(selector)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var device = AudioDeviceID(kAudioObjectUnknown)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &device
        )
        guard status == noErr, device != kAudioObjectUnknown else {
            throw CaptureError("no default \(label) device (OSStatus \(status))")
        }
        return device
    }

    private static func stringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) throws -> String {
        var address = globalAddress(selector)
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var value: Unmanaged<CFString>?
        let status = AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value)
        guard status == noErr, let value else {
            throw CaptureError("could not read property \(selector) (OSStatus \(status))")
        }
        return value.takeRetainedValue() as String
    }

    private static func globalAddress(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
    }
}

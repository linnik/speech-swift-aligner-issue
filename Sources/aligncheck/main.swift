import AudioCommon
import CoreML
import Foundation
import Qwen3ASR

// Usage: aligncheck <model dir> <wav file> <text> [default|all|cpu|gpu]
//
// default  the compute units CoreMLForcedAligner picks on its own
// all      .all for the encoder and the decoder
// cpu      .cpuOnly for both
// gpu      .cpuAndGPU for both
//
// Run with ALIGN_DEBUG=1 to also print the raw timestamp indices.
let arguments = CommandLine.arguments
let modelDirectory = URL(fileURLWithPath: arguments[1], isDirectory: true)
let audioFile = URL(fileURLWithPath: arguments[2])
let text = arguments[3]
let mode = arguments.count > 4 ? arguments[4] : "default"

let aligner: CoreMLForcedAligner
if mode == "default" {
    aligner = try CoreMLForcedAligner.fromDirectory(modelDirectory)
} else {
    let unitsByMode: [String: MLComputeUnits] = [
        "all": .all, "cpu": .cpuOnly, "gpu": .cpuAndGPU, "ne": .cpuAndNeuralEngine,
    ]
    guard let computeUnits = unitsByMode[mode] else {
        fatalError("unknown mode \(mode); use default, all, cpu, gpu or ne")
    }
    aligner = try CoreMLForcedAligner.fromDirectory(
        modelDirectory,
        encoderComputeUnits: computeUnits,
        decoderComputeUnits: computeUnits
    )
}

// The aligner takes 16 kHz audio.
let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: audioFile)
let audio: [Float]
if sampleRate == 16000 {
    audio = samples
} else {
    audio = AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
}

let words = try aligner.align(audio: audio, text: text, sampleRate: 16000, language: "English")
for word in words {
    print(String(format: "%6.2f  %@", word.startTime, word.text))
}

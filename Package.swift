// swift-tools-version: 5.10
import PackageDescription

// Expects a speech-swift checkout next to this folder, at ../speech-swift.
// Check out the commit you want to test there, then build this package.
let package = Package(
    name: "aligncheck",
    platforms: [.macOS("15.0")],
    dependencies: [.package(path: "../speech-swift")],
    targets: [
        .executableTarget(
            name: "aligncheck",
            dependencies: [
                .product(name: "Qwen3ASR", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift"),
            ])
    ]
)

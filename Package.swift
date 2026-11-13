// swift-tools-version:5.9
import PackageDescription

// The app is built by XcodeGen (project.yml). This package only exists to test
// the parts of it that need no iPhone: `swift test` on any Mac, and in CI.
let package = Package(
    name: "LocalModelsLogic",
    platforms: [.macOS(.v14), .iOS(.v17)],
    targets: [
        .target(name: "LocalModelsLogic", path: "App/Logic"),
        .testTarget(name: "LocalModelsLogicTests", dependencies: ["LocalModelsLogic"], path: "Tests/LogicTests"),
    ]
)

import SwiftUI
import UIKit

@main
struct LocalModelsApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var store = ModelStore.shared
    @State private var chat = ChatViewModel()

    init() {
        AppSettings.register()
    }

    var body: some Scene {
        WindowGroup {
            ChatView()
                .environment(store)
                .environment(chat)
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        ModelStore.shared.backgroundCompletionHandler = completionHandler
    }
}

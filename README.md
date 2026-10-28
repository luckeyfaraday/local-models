# Local Models

An iPhone app that runs open-weight language models on the phone itself. It wraps
[llama.cpp](https://github.com/ggml-org/llama.cpp) on the GPU (Metal), so it loads any GGUF the
pinned llama.cpp build supports, including models released yesterday.

- Download any `.gguf` from a public Hugging Face repo inside the app, or copy one in through
  Files › On My iPhone › Local Models.
- Streaming chat with tokens/sec, reasoning on/off for thinking models (LFM3, Qwen-style),
  sampling presets, context length and a system prompt.
- Nothing leaves the phone except the model download.

The default suggestion is Liquid AI's [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF)
(about 1 GB at Q4_0), which needs llama.cpp `b11960` or newer.

## Install

There's no App Store build. CI builds an **unsigned** IPA; you sign it with your own Apple ID.

1. Get `LocalModels-unsigned.ipa` from the latest [release](../../releases) or CI run.
2. Sign and install it with any sideloader: Impactor/`plumesign`, AltStore, SideStore, or your own
   developer certificate. With `plumesign` logged in and the iPhone on USB:

   ```bash
   BUNDLE_ID=com.yourname.localmodels scripts/install.sh LocalModels-unsigned.ipa
   ```

3. On the phone: Settings › General › VPN & Device Management › trust your developer certificate.

Free Apple IDs sign apps for 7 days, so re-run the install before then. Your downloaded models
stay on the phone across reinstalls.

## Updating llama.cpp

`LLAMA_VERSION` pins the llama.cpp release and the SHA-256 of its `xcframework.zip`.

```bash
scripts/bump-llama.sh          # newest release, or: scripts/bump-llama.sh b12345
git commit -am "Bump llama.cpp" && git push   # CI builds a new IPA
```

## Building

CI (`.github/workflows/build.yml`) runs on a GitHub macOS runner:
`scripts/fetch-llama.sh` → `xcodegen generate` → `xcodebuild` without signing → `dist/LocalModels-unsigned.ipa`.
Tagging `v*` attaches the IPA to a release. On a Mac you can run the same steps and open
`LocalModels.xcodeproj`.

## Limits

- iOS stops GPU work in the background, so a reply stops when you leave the app. Downloads keep
  going.
- Every turn re-reads the whole chat, since hybrid models such as LFM can't trim part of their
  state. Prefill is fast on the GPU, but very long chats get slower to start.
- Text only for now: no vision, audio or tool calling.

## License

MIT. llama.cpp is MIT and is downloaded at build time, not vendored. Models keep their own licences.

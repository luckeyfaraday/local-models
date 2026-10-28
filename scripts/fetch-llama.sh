#!/usr/bin/env bash
# Downloads the pinned llama.cpp iOS framework (see LLAMA_VERSION) into Vendor/.
set -euo pipefail
cd "$(dirname "$0")/.."
read -r tag sha < LLAMA_VERSION
if [ -f Vendor/llama.xcframework/.tag ] && [ "$(cat Vendor/llama.xcframework/.tag)" = "$tag" ]; then
  echo "llama.cpp $tag already in Vendor/"
  exit 0
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/llama.zip" "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-xcframework.zip"
echo "$sha  $tmp/llama.zip" | shasum -a 256 -c -
unzip -q "$tmp/llama.zip" -d "$tmp/x"
rm -rf Vendor/llama.xcframework
mkdir -p Vendor
mv "$tmp/x/build-apple/llama.xcframework" Vendor/
echo "$tag" > Vendor/llama.xcframework/.tag
echo "llama.cpp $tag -> Vendor/llama.xcframework"

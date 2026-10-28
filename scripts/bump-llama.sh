#!/usr/bin/env bash
# Points LLAMA_VERSION at a llama.cpp release (default: the newest one with an xcframework).
# usage: scripts/bump-llama.sh [b12345]
set -euo pipefail
cd "$(dirname "$0")/.."
tag=${1:-$(gh api repos/ggml-org/llama.cpp/releases --jq '[.[] | select(.assets[].name | endswith("-xcframework.zip"))][0].tag_name')}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/llama.zip" "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-xcframework.zip"
sha=$(shasum -a 256 "$tmp/llama.zip" | cut -d' ' -f1)
echo "$tag $sha" > LLAMA_VERSION
echo "LLAMA_VERSION -> $tag ($sha)"

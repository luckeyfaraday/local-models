#!/usr/bin/env bash
# Signs the unsigned IPA with your Apple ID and installs it on an iPhone plugged in over USB.
# Needs plumesign (the CLI from Impactor) already logged in: `plumesign account login`.
#
# usage: scripts/install.sh [LocalModels-unsigned.ipa]   (default: newest green CI build of main)
# env:   PLUMESIGN  path to plumesign (default: plumesign on PATH)
#        BUNDLE_ID  app id to register; free Apple IDs need one nobody else has taken
set -euo pipefail
cd "$(dirname "$0")/.."
PLUMESIGN=${PLUMESIGN:-plumesign}
BUNDLE_ID=${BUNDLE_ID:-io.github.luckeyfaraday.localmodels}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

if [ $# -ge 1 ]; then
  ipa=$1
else
  run=$(gh run list --workflow build --branch main --status success --limit 1 --json databaseId --jq '.[0].databaseId')
  [ -n "$run" ] || { echo "no successful CI build of main yet" >&2; exit 1; }
  gh run download "$run" --name LocalModels-unsigned-ipa --dir "$work"
  ipa=$work/LocalModels-unsigned.ipa
fi
unzip -q "$ipa" -d "$work/ipa"

# plumesign's --udid wants usbmuxd's numeric DeviceID, not the UDID.
mux_id=$(python3 - <<'PY'
import plistlib, socket, struct
s = socket.socket(socket.AF_UNIX)
s.connect("/var/run/usbmuxd")
body = plistlib.dumps({"MessageType": "ListDevices", "ClientVersionString": "x", "ProgName": "x"})
s.sendall(struct.pack("<IIII", 16 + len(body), 1, 8, 1) + body)
n = struct.unpack("<I", s.recv(16)[:4])[0] - 16
data = b""
while len(data) < n:
    data += s.recv(n - len(data))
usb = [d["DeviceID"] for d in plistlib.loads(data)["DeviceList"] if d["Properties"].get("ConnectionType") == "USB"]
print(usb[0] if usb else "")
PY
)
[ -n "$mux_id" ] || { echo "no iPhone on USB" >&2; exit 1; }

"$PLUMESIGN" sign -p "$work/ipa/Payload/LocalModels.app" --apple-id --custom-identifier "$BUNDLE_ID" \
  --register-and-install --udid "$mux_id" < /dev/null
echo "Installed. First launch from a new developer: Settings > General > VPN & Device Management > Trust."

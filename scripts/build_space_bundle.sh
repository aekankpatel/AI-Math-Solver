#!/usr/bin/env bash
# Assemble the Hugging Face Space repository content into a directory (default: ./space-bundle).
# Contents: the Space README (with its config header), its Dockerfile, backend/app and backend/requirements.txt.
# Nothing else from the repository (no tests, benchmark data, frontend or local files) is included.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$root/space-bundle}"
rm -rf "$out"
mkdir -p "$out"
cp "$root/deploy/huggingface/README.md" "$root/deploy/huggingface/Dockerfile" "$out/"
cp "$root/backend/requirements.txt" "$out/"
cp -r "$root/backend/app" "$out/app"
find "$out" \( -name '__pycache__' -o -name '*.pyc' -o -name '.env' -o -name '*.jsonl' \) -prune -exec rm -rf {} +
echo "Space bundle written to $out"

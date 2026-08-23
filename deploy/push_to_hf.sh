#!/usr/bin/env bash
# Publish the demo to a Hugging Face Space.
#
#   1. Get a WRITE token: https://huggingface.co/settings/tokens
#   2. export HF_TOKEN=hf_...
#   3. ./deploy/push_to_hf.sh <your-hf-username>
set -euo pipefail

USER="${1:?usage: ./deploy/push_to_hf.sh <hf-username>}"
SPACE="${2:-allocation-agent}"
: "${HF_TOKEN:?set HF_TOKEN first: export HF_TOKEN=hf_...}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT/deploy/SPACE_README.md" "$STAGE/README.md"
cp "$ROOT/Dockerfile" "$ROOT/pyproject.toml" "$STAGE/"
cp -r "$ROOT/src" "$ROOT/artifacts" "$STAGE/"
rm -f "$STAGE/artifacts/runs.db"          # runtime state, not an artifact

echo "staged $(du -sh "$STAGE" | cut -f1) for $USER/$SPACE"
python3 - "$USER" "$SPACE" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi
user, space, stage = sys.argv[1:4]
api = HfApi()
repo = f"{user}/{space}"
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(folder_path=stage, repo_id=repo, repo_type="space",
                  commit_message="deploy allocation agent")
print(f"\n  https://huggingface.co/spaces/{repo}")
print("  first build takes ~3-5 minutes")
PY

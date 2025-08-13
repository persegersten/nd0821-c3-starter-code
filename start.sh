#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_TOKEN:?GITHUB_TOKEN not set}"
: "${GITHUB_OWNER:?Set GITHUB_OWNER in Heroku Config Vars}"
: "${GITHUB_REPO:?Set GITHUB_REPO in Heroku Config Vars}"
: "${PORT:?PORT not set}"
: "${WEB_CONCURRENCY:?WEB_CONCURRENCY not set}"
: "${MALLOC_ARENA_MAX:?MALLOC_ARENA_MAX not set}"
ARTIFACT_NAME="${ARTIFACT_NAME:-model}"

mkdir -p model /tmp/artifact
echo "Looking for latest non-expired artifact named '${ARTIFACT_NAME}' in ${GITHUB_OWNER}/${GITHUB_REPO} ..."

# 1) Hämta artifact_id för senaste icke-expired med rätt namn
AID="$(python - <<'PY'
import os, json, urllib.request, sys
owner=os.environ["GITHUB_OWNER"]; repo=os.environ["GITHUB_REPO"]; name=os.environ.get("ARTIFACT_NAME","model")
r=urllib.request.Request(f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts?per_page=100")
r.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
r.add_header("X-GitHub-Api-Version", "2022-11-28")
with urllib.request.urlopen(r) as resp:
    data=json.load(resp)
arts=[a for a in data.get("artifacts",[]) if a.get("name")==name and not a.get("expired")]
if not arts:
    print(f"No non-expired artifact named '{name}' found", file=sys.stderr); sys.exit(2)
aid=sorted(arts, key=lambda a:a["created_at"], reverse=True)[0]["id"]
print(aid)
PY
)"

echo "Using artifact id: $AID"

# 2) Ladda ner zip via GitHub API (curl -L släpper Authorization vid cross-host redirect)
ZIP_URL="https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/artifacts/${AID}/zip"
echo "Downloading ZIP from: $ZIP_URL"
curl -sS -f -L \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -o /tmp/artifact/artifact.zip \
  "$ZIP_URL"

# 3) Snabb sanity: är det verkligen en zip?
unzip -t /tmp/artifact/artifact.zip >/dev/null

# 4) Extrahera bara de filer vi behöver
unzip -o /tmp/artifact/artifact.zip \
  "*/random_forest_model.joblib" "*/onehot_encoder.joblib" -d /tmp/artifact >/dev/null

# Flytta till ./model (flatten)
find /tmp/artifact -type f -name "random_forest_model.joblib" -exec cp {} model/ \;
find /tmp/artifact -type f -name "onehot_encoder.joblib"   -exec cp {} model/ \;

# 5) Verifiera att båda finns
test -f model/random_forest_model.joblib || { echo "Missing random_forest_model.joblib"; exit 4; }
test -f model/onehot_encoder.joblib     || { echo "Missing onehot_encoder.joblib"; exit 4; }

echo "Artifact downloaded and extracted."

# 6) Starta appen (bind till Herokus PORT)
exec gunicorn components.api.app:app \
  --workers "${WEB_CONCURRENCY}" \
  --max-requests 200 --max-requests-jitter 50 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT

#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_TOKEN:?GITHUB_TOKEN not set}"
: "${PORT:?PORT not set}"
: "${GITHUB_OWNER:?Set GITHUB_OWNER in Heroku Config Vars}"
: "${GITHUB_REPO:?Set GITHUB_REPO in Heroku Config Vars}"
: "${ARTIFACT_NAME:=model}"   # default 'model' om inte satt

mkdir -p model

echo "Downloading artifact '${ARTIFACT_NAME}' from ${GITHUB_OWNER}/${GITHUB_REPO} ..."

python - <<'PY'
import os, sys, json, io, zipfile, urllib.request

owner = os.environ["GITHUB_OWNER"]
repo  = os.environ["GITHUB_REPO"]
name  = os.environ.get("ARTIFACT_NAME","model")
tok   = os.environ["GITHUB_TOKEN"]

def req(url, accept=None):
    r = urllib.request.Request(url)
    r.add_header("Authorization", f"Bearer {tok}")
    r.add_header("X-GitHub-Api-Version", "2022-11-28")
    if accept:
        r.add_header("Accept", accept)
    return urllib.request.urlopen(r)

# 1) list artifacts
with req(f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts?per_page=100",
         "application/vnd.github+json") as resp:
    data = json.load(resp)

arts = [a for a in data.get("artifacts", []) if a.get("name") == name and not a.get("expired")]
if not arts:
    print(f"No non-expired artifact named '{name}' found in {owner}/{repo}", file=sys.stderr)
    sys.exit(2)

aid = sorted(arts, key=lambda a: a["created_at"], reverse=True)[0]["id"]

# 2) download zip (GitHub returns a 302 to a signed URL; urllib follows it)
with req(f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts/{aid}/zip") as resp:
    zbytes = resp.read()

# 3) validate & extract exactly the files we expect
try:
    zf = zipfile.ZipFile(io.BytesIO(zbytes))
except zipfile.BadZipFile:
    print("Downloaded file is not a zip. First 400 bytes:", zbytes[:400], file=sys.stderr)
    sys.exit(3)

wanted = {"random_forest_model.joblib", "onehot_encoder.joblib"}
extracted = set()
import os as _os
_os.makedirs("model", exist_ok=True)

for m in zf.infolist():
    base = _os.path.basename(m.filename)
    if base in wanted:
        with zf.open(m) as src, open(_os.path.join("model", base), "wb") as dst:
            dst.write(src.read())
        extracted.add(base)

missing = wanted - extracted
if missing:
    print("Missing expected files in artifact:", ", ".join(sorted(missing)), file=sys.stderr)
    sys.exit(4)

print("Artifact downloaded and extracted.")
PY

ls -la model

# Start the app (justera workers om du vill)
exec gunicorn components.api.app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT

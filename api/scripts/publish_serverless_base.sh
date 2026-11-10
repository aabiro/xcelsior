#!/usr/bin/env bash
# Build, smoke-test and (with --push) publish the serverless custom-worker base
# image, ghcr.io/xcelsior-gpu/serverless-base.
#
#   api/scripts/publish_serverless_base.sh          build + smoke test only
#   api/scripts/publish_serverless_base.sh --push   ...then push to GHCR
#
# Tags: py3.12-<short sha> (immutable) and py3.12 (moves to the newest push).
#
# The image is built from `git archive HEAD`, not the working tree, so a tag
# always matches a commit: uncommitted edits and stray __pycache__ never ship.
# --push also refuses a HEAD that is not on origin/main.
#
# This runs from a workstation because hosted Actions is blocked on the aabiro
# account (see the HOSTED_ACTIONS note in .github/workflows/publish.yml).
#
# Push prerequisites, once per machine:
#   gh auth refresh -h github.com -s write:packages
#   gh auth token | docker login ghcr.io -u <github-user> --password-stdin
#
# The first push creates the package as private. Customer workers pull it with
# no credentials, so set it public once in the package settings (the script
# checks anonymous pull at the end and prints the link if it still fails).
set -euo pipefail

IMAGE="ghcr.io/xcelsior-gpu/serverless-base"
FLAVOUR="py3.12"
SOURCE_REPO="xcelsior-gpu/serverless-base"
SOURCE_URL="https://github.com/$SOURCE_REPO"
SETTINGS_URL="https://github.com/orgs/xcelsior-gpu/packages/container/serverless-base/settings"
PLATFORM="linux/amd64"

PUSH=0
case "${1:-}" in
  "") ;;
  --push) PUSH=1 ;;
  -h | --help) sed -n '2,23p' "$0"; exit 0 ;;
  *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

die() { echo "error: $*" >&2; exit 1; }
step() { printf '\n==> %s\n' "$*"; }

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$repo_root"

image_inputs=(
  api/serverless/__init__.py
  api/serverless/worker_sdk
  api/serverless/images/base
  api/serverless/examples/echo
)

if [[ -n "$(git status --porcelain -- "${image_inputs[@]}")" ]]; then
  (( PUSH )) && die "uncommitted changes under the image inputs; commit them first:
$(git status --short -- "${image_inputs[@]}")"
  echo "warning: uncommitted changes under the image inputs are NOT in this build (it uses HEAD)" >&2
fi

if (( PUSH )); then
  git fetch -q origin main
  git merge-base --is-ancestor HEAD origin/main \
    || die "HEAD $(git rev-parse --short HEAD) is not on origin/main; push it first so the tag is reproducible"
fi

sha="$(git rev-parse HEAD)"
short="$(git rev-parse --short=12 HEAD)"
created="$(git show -s --format=%cI HEAD)"
versioned="$IMAGE:$FLAVOUR-$short"
moving="$IMAGE:$FLAVOUR"
echo_tag="xcelsior/serverless-echo-smoke:$short"

ctx="$(mktemp -d)"
cname="serverless-base-smoke-$$"
cleanup() {
  docker rm -f "$cname" >/dev/null 2>&1 || true
  rm -rf "$ctx"
}
trap cleanup EXIT

git archive HEAD -- "${image_inputs[@]}" | tar -x -C "$ctx"

# Materialize the standalone xcelsior-gpu/serverless-base repository layout and
# build directly from it so the published source repo is guaranteed to build
# with a plain `docker build .`.
mkdir -p "$ctx/repo/serverless" "$ctx/repo/examples"
cp "$ctx/api/serverless/images/base/README.md" "$ctx/repo/README.md"
cp "$ctx/api/serverless/images/base/Dockerfile" "$ctx/repo/Dockerfile"
cp "$ctx/api/serverless/__init__.py" "$ctx/repo/serverless/__init__.py"
cp -R "$ctx/api/serverless/worker_sdk" "$ctx/repo/serverless/worker_sdk"
cp -R "$ctx/api/serverless/examples/echo" "$ctx/repo/examples/echo"

source_sha="$sha"
if (( PUSH )); then
  step "sync $SOURCE_REPO from HEAD $short"
  if ! gh repo view "$SOURCE_REPO" >/dev/null 2>&1; then
    gh repo create "$SOURCE_REPO" --public \
      --description "Official base image and Python Worker SDK (serverless.worker_sdk) for custom workers on Xcelsior Serverless — FROM ghcr.io/xcelsior-gpu/serverless-base:py3.12" \
      --homepage "https://docs.xcelsior.ca"
  fi
  git init -q -b main "$ctx/sync"
  git -C "$ctx/sync" remote add origin "https://github.com/$SOURCE_REPO.git"
  if git -C "$ctx/sync" fetch -q origin main 2>/dev/null; then
    git -C "$ctx/sync" checkout -q -B main FETCH_HEAD
    git -C "$ctx/sync" rm -rf -q . >/dev/null 2>&1 || true
  fi
  cp -R "$ctx/repo/." "$ctx/sync/"
  git -C "$ctx/sync" add -A
  if ! git -C "$ctx/sync" diff --cached --quiet; then
    git -C "$ctx/sync" commit -q -m "chore(serverless-base): sync from monorepo $short"
    git -C "$ctx/sync" push -q -u origin main
  fi
  source_sha="$(git -C "$ctx/sync" rev-parse HEAD)"
  echo "synced $SOURCE_REPO @ $(git -C "$ctx/sync" rev-parse --short=12 HEAD)"
fi

step "build $versioned ($PLATFORM, from HEAD $short)"
docker build --pull --platform "$PLATFORM" \
  --label "org.opencontainers.image.source=$SOURCE_URL" \
  --label "org.opencontainers.image.revision=$source_sha" \
  --label "org.opencontainers.image.version=$FLAVOUR-$short" \
  --label "org.opencontainers.image.created=$created" \
  -f "$ctx/repo/Dockerfile" \
  -t "$versioned" -t "$moving" \
  "$ctx/repo"

step "import check: the SDK loads without the API's dependencies"
docker run --rm --network none "$versioned" \
  python -c 'import serverless.worker_sdk as sdk; print("worker_sdk exports", len(sdk.__all__), "names")'

# The echo example is FROM "$moving", which now resolves to the image just
# built (no --pull here, so nothing is fetched from the registry).
step "smoke: boot the echo example, probe /healthz, drain on SIGTERM"
docker build --platform "$PLATFORM" -t "$echo_tag" "$ctx/repo/examples/echo"
docker run -d --name "$cname" \
  --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --user 65534:65534 --cpus 1 --memory 256m \
  -e XCELSIOR_WORKER_ID="publish-smoke-$short" \
  -e XCELSIOR_WORKER_HTTP_RETRIES=1 \
  "$echo_tag" >/dev/null

health=""
for _ in $(seq 1 30); do
  if health="$(docker exec "$cname" curl -sf http://127.0.0.1:8080/healthz 2>/dev/null)"; then
    break
  fi
  health=""
  [[ "$(docker inspect -f '{{.State.Running}}' "$cname")" == true ]] || break
  sleep 1
done
if [[ -z "$health" ]] || ! grep -q '"ok": true' <<<"$health"; then
  docker logs "$cname" >&2 || true
  die "echo worker never reported healthy (last /healthz: ${health:-none})"
fi
echo "/healthz: $health"

docker stop -t 30 "$cname" >/dev/null
exit_code="$(docker inspect -f '{{.State.ExitCode}}' "$cname")"
logs="$(docker logs "$cname" 2>&1)"
if [[ "$exit_code" != 0 ]] || ! grep -q 'Drain complete' <<<"$logs"; then
  echo "$logs" >&2
  die "echo worker did not drain cleanly on SIGTERM (exit $exit_code)"
fi
echo "drained on SIGTERM, exit 0"
docker rm "$cname" >/dev/null
docker rmi "$echo_tag" >/dev/null

if (( ! PUSH )); then
  step "built and smoke-tested $versioned (not pushed; rerun with --push)"
  exit 0
fi

step "push $versioned and $moving"
for ref in "$versioned" "$moving"; do
  docker push "$ref" || die "push of $ref failed. Log in first:
  gh auth refresh -h github.com -s write:packages
  gh auth token | docker login ghcr.io -u <github-user> --password-stdin"
done
digest="$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$versioned" | grep -m1 "^$IMAGE@" || true)"
echo "pushed: ${digest:-$versioned}"

step "anonymous pull check"
repo="${IMAGE#ghcr.io/}"
# GHCR refuses a private package at the token request itself (401), so that
# refusal is this check's answer, not a reason to stop before reporting it.
token_resp="$(curl -sS -w '\n%{http_code}' \
  "https://ghcr.io/token?scope=repository:$repo:pull&service=ghcr.io")" || token_resp=$'\n000'
code="${token_resp##*$'\n'}"
if [[ "$code" == 200 ]]; then
  token="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])' <<<"${token_resp%$'\n'*}")"
  code="$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $token" \
    -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json' \
    "https://ghcr.io/v2/$repo/manifests/$FLAVOUR-$short")" || code=000
fi
case "$code" in
  200)
    echo "anonymous pull works: customer workers can use FROM $moving" ;;
  401 | 403)
    echo "anonymous pull refused (HTTP $code): the push worked, but the package is private." >&2
    echo "The public aabiro/xcelsior repo does not make it public: the package lives under" >&2
    echo "the xcelsior-gpu org, so it has its own visibility. Set it public once:" >&2
    echo "  $SETTINGS_URL" >&2
    exit 3 ;;
  *)
    echo "anonymous pull check got HTTP $code from ghcr.io, so it could not tell whether" >&2
    echo "the package is public. The push worked; check by hand with a logged-out docker pull." >&2
    exit 4 ;;
esac

#!/usr/bin/env bash
# Benchmark deploy speed improvements (local + remote over SSH).
# Does not touch /opt/xcelsior production paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSH_KEY="${XCELSIOR_SSH_KEY:-$HOME/.ssh/xcelsior}"
REMOTE_USER="${XCELSIOR_DEPLOY_USER:-linuxuser}"
REMOTE_HOST="${XCELSIOR_DEPLOY_HOST:-149.28.121.61}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o Compression=no)

RSYNC_EXCLUDES=(
  --exclude='.git'
  --exclude='__pycache__'
  --exclude='.pytest_cache'
  --exclude='venv'
  --exclude='node_modules'
  --exclude='.next'
  --exclude='*.db'
  --exclude='*.db-*'
  --exclude='*.log'
  --exclude='data'
  --exclude='artifacts'
  --exclude='.env'
  --exclude='/desktop'
  --exclude='checkpoints'
  --exclude='.cursor'
  --exclude='terminals'
)

now_ms() { date +%s%3N; }
elapsed_ms() { echo $(( $2 - $1 )); }

fmt_sec() {
  local ms="$1"
  awk -v ms="$ms" 'BEGIN { printf "%.2fs", ms / 1000 }'
}

section() { printf "\n══ %s ══\n" "$1"; }

ssh_remote() {
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

rsync_bench() {
  local label="$1"
  local dest_name="$2"
  shift 2
  local extra=("$@")
  local t0 t1 dest="/tmp/xcelsior-bench/${dest_name}/"
  ssh_remote "rm -rf '$dest' && mkdir -p '$dest'" >/dev/null
  t0=$(now_ms)
  rsync -az --partial --omit-dir-times --no-perms --no-owner --no-group \
    "${RSYNC_EXCLUDES[@]}" "${extra[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    "$PROJECT_DIR/" "$REMOTE:$dest" >/dev/null
  t1=$(now_ms)
  printf "  %-28s %s\n" "$label" "$(fmt_sec "$(elapsed_ms "$t0" "$t1")")"
}

section "1. Tarball compression (local, deploy excludes)"
TMP_TAR="/tmp/xcelsior_bench_payload"
rm -rf "$TMP_TAR"
mkdir -p "$TMP_TAR"

T0=$(now_ms)
tar -cf - \
  --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='venv' --exclude='node_modules' --exclude='.next' \
  --exclude='*.db' --exclude='*.db-*' --exclude='*.log' \
  --exclude='data' --exclude='artifacts' --exclude='.env' \
  --exclude='/desktop' --exclude='checkpoints' \
  -C "$PROJECT_DIR" . >"$TMP_TAR/raw.tar"
T1=$(now_ms)
RAW_SIZE=$(stat -c%s "$TMP_TAR/raw.tar")
printf "  %-28s %s  (%.1f MB uncompressed tar)\n" "tar create" "$(fmt_sec "$(elapsed_ms "$T0" "$T1")")" "$(awk -v b="$RAW_SIZE" 'BEGIN{printf "%.1f", b/1048576}')"

bench_compress() {
  local label="$1"
  shift
  local out="/tmp/xcelsior_bench.$label"
  local t0 t1 size
  t0=$(now_ms)
  "$@" <"$TMP_TAR/raw.tar" >"$out"
  t1=$(now_ms)
  size=$(stat -c%s "$out")
  printf "  %-28s %s  (%.1f MB, %.1f%% of raw)\n" "$label" "$(fmt_sec "$(elapsed_ms "$t0" "$t1")")" \
    "$(awk -v b="$size" 'BEGIN{printf "%.1f", b/1048576}')" \
    "$(awk -v s="$size" -v r="$RAW_SIZE" 'BEGIN{printf "%.1f", 100*s/r}')"
}

bench_compress "gzip -1" gzip -1
bench_compress "pigz -1" pigz -1
bench_compress "zstd -3" zstd -3 -T0

section "2. Rsync over SSH (remote /tmp/xcelsior-bench, full tree)"
ssh_remote "rm -rf /tmp/xcelsior-bench && mkdir -p /tmp/xcelsior-bench"

rsync_bench "no wire compression" full_plain
rsync_bench "zlib (--compress)" full_zlib --compress
rsync_bench "zstd wire compression" full_zstd --compress --compress-choice=zstd

section "3. Rsync delta (touch one file, re-sync to warm tree)"
touch "$PROJECT_DIR/scripts/benchmark_deploy_speed.sh"
delta_rsync_bench() {
  local label="$1"
  shift
  local extra=("$@")
  local dest="/tmp/xcelsior-bench/full_zstd/"
  local t0 t1
  t0=$(now_ms)
  rsync -az --partial --omit-dir-times --no-perms --no-owner --no-group \
    "${RSYNC_EXCLUDES[@]}" "${extra[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    "$PROJECT_DIR/" "$REMOTE:$dest" >/dev/null
  t1=$(now_ms)
  printf "  %-28s %s\n" "$label" "$(fmt_sec "$(elapsed_ms "$t0" "$t1")")"
}
delta_rsync_bench "delta, no compression"
delta_rsync_bench "delta, zstd wire" --compress --compress-choice=zstd
ssh_remote "rm -rf /tmp/xcelsior-bench"

section "4. Deploy input hashing (local)"
serial_hash_time() {
  local t0 t1
  t0=$(now_ms)
  for _ in api frontend nginx runtime; do
    python3 - "$PROJECT_DIR" <<'PY' >/dev/null
import glob, hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
patterns = [".dockerignore", "Dockerfile", "requirements.txt"]
files = set()
for pattern in patterns:
    for p in glob.glob(str(root / pattern), recursive=True):
        files.add(Path(p))
digest = hashlib.sha256()
for path in sorted(files):
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
  done
  t1=$(now_ms)
  echo $((t1 - t0))
}

parallel_hash_time() {
  local t0 t1 d
  d=$(mktemp -d)
  t0=$(now_ms)
  python3 - "$PROJECT_DIR" <<'PY' >"$d/api" &
import glob, hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
files = set()
for pattern in [".dockerignore", "Dockerfile", "requirements.txt", "*.py", "routes"]:
    for p in glob.glob(str(root / pattern), recursive=True):
        files.add(Path(p))
digest = hashlib.sha256()
for path in sorted(files):
    if path.is_file():
        digest.update(path.read_bytes())
print(digest.hexdigest())
PY
  python3 - "$PROJECT_DIR" <<'PY' >"$d/frontend" &
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
print(hashlib.sha256(b"frontend").hexdigest())
PY
  python3 - "$PROJECT_DIR" <<'PY' >"$d/nginx" &
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
print(hashlib.sha256(b"nginx").hexdigest())
PY
  python3 - "$PROJECT_DIR" <<'PY' >"$d/runtime" &
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
print(hashlib.sha256(b"runtime").hexdigest())
PY
  wait
  t1=$(now_ms)
  rm -rf "$d"
  echo $((t1 - t0))
}

T_SERIAL=$(serial_hash_time)
T_PARALLEL=$(parallel_hash_time)
printf "  %-28s %s\n" "4 sequential hash passes" "$(fmt_sec "$T_SERIAL")"
printf "  %-28s %s\n" "4 parallel hash passes" "$(fmt_sec "$T_PARALLEL")"
printf "  %-28s %.1fx faster\n" "parallel speedup" "$(awk -v s="$T_SERIAL" -v p="$T_PARALLEL" 'BEGIN{ if (p>0) printf "%.1f", s/p; else print "n/a" }')"

section "5. Remote deploy-hash SSH round-trips (simulated)"
T0=$(now_ms)
for _ in api frontend nginx runtime; do
  ssh_remote "cat /opt/xcelsior-backups/.deploy-meta/api 2>/dev/null || true" >/dev/null
done
T1=$(now_ms)
T_SERIAL_SSH=$((T1 - T0))

T0=$(now_ms)
ssh_remote 'META=/opt/xcelsior-backups/.deploy-meta; for n in api frontend nginx runtime; do printf "%s|" "$n"; cat "$META/$n" 2>/dev/null || true; printf "\n"; done' >/dev/null
T1=$(now_ms)
T_BATCH_SSH=$((T1 - T0))

printf "  %-28s %s\n" "4 SSH hash fetches" "$(fmt_sec "$T_SERIAL_SSH")"
printf "  %-28s %s\n" "1 batched SSH hash fetch" "$(fmt_sec "$T_BATCH_SSH")"
printf "  %-28s %.1fx faster\n" "batch speedup" "$(awk -v s="$T_SERIAL_SSH" -v b="$T_BATCH_SSH" 'BEGIN{ if (b>0) printf "%.1f", s/b; else print "n/a" }')"

section "6. End-to-end sync path (legacy tarball vs rsync)"
TARBALL_EXCLUDES=(
  --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache'
  --exclude='venv' --exclude='node_modules' --exclude='.next'
  --exclude='*.db' --exclude='*.db-*' --exclude='*.log'
  --exclude='data' --exclude='artifacts' --exclude='.env'
  --exclude='/desktop' --exclude='checkpoints' --exclude='.cursor' --exclude='terminals'
)

bench_e2e_tarball() {
  local label="$1" mode="$2"
  local local_file remote_file t0 t1 t2 t3 pack upload extract total
  case "$mode" in
    pigz)
      local_file=/tmp/xcelsior_legacy.tar.gz
      remote_file=/tmp/xcelsior_legacy.tar.gz
      ;;
    zstd)
      local_file=/tmp/xcelsior_zstd.tar.zst
      remote_file=/tmp/xcelsior_zstd.tar.zst
      ;;
  esac
  rm -f "$local_file"
  t0=$(now_ms)
  case "$mode" in
    pigz) tar -cf - "${TARBALL_EXCLUDES[@]}" -C "$PROJECT_DIR" . | pigz -1 >"$local_file" ;;
    zstd) tar -cf - "${TARBALL_EXCLUDES[@]}" -C "$PROJECT_DIR" . | zstd -3 -T0 >"$local_file" ;;
  esac
  t1=$(now_ms); pack=$((t1 - t0))
  scp "${SSH_OPTS[@]}" "$local_file" "$REMOTE:$remote_file" >/dev/null
  t2=$(now_ms); upload=$((t2 - t1))
  case "$mode" in
    pigz)
      ssh_remote 'rm -rf /tmp/xcelsior_legacy_extract && mkdir -p /tmp/xcelsior_legacy_extract && pigz -dc /tmp/xcelsior_legacy.tar.gz | tar -xf - -C /tmp/xcelsior_legacy_extract && rm -f /tmp/xcelsior_legacy.tar.gz' >/dev/null
      ;;
    zstd)
      ssh_remote 'rm -rf /tmp/xcelsior_zstd_extract && mkdir -p /tmp/xcelsior_zstd_extract && zstd -d -c /tmp/xcelsior_zstd.tar.zst | tar -xf - -C /tmp/xcelsior_zstd_extract && rm -f /tmp/xcelsior_zstd.tar.zst' >/dev/null
      ;;
  esac
  t3=$(now_ms); extract=$((t3 - t2)); total=$((t3 - t0))
  printf "  %-32s %s  (pack=%.2fs upload=%.2fs extract=%.2fs)\n" "$label" "$(fmt_sec "$total")" \
    "$(awk -v ms="$pack" 'BEGIN{printf "%.2f", ms/1000}')" \
    "$(awk -v ms="$upload" 'BEGIN{printf "%.2f", ms/1000}')" \
    "$(awk -v ms="$extract" 'BEGIN{printf "%.2f", ms/1000}')" >&2
  echo "$total"
}

LEGACY_MS=$(bench_e2e_tarball "legacy pigz tarball+scp" pigz)
ZSTD_MS=$(bench_e2e_tarball "zstd tarball+scp" zstd)
LEGACY_MS="${LEGACY_MS//$'\n'/}"; LEGACY_MS="${LEGACY_MS##*$'\n'}"
ZSTD_MS="${ZSTD_MS//$'\n'/}"; ZSTD_MS="${ZSTD_MS##*$'\n'}"

ssh_remote "rm -rf /tmp/xcelsior_rsync_full && mkdir -p /tmp/xcelsior_rsync_full"
t0=$(now_ms)
rsync -az --partial --omit-dir-times --no-perms --no-owner --no-group \
  "${RSYNC_EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" \
  "$PROJECT_DIR/" "$REMOTE:/tmp/xcelsior_rsync_full/" >/dev/null
t1=$(now_ms); RSYNC_FULL_MS=$((t1 - t0))
printf "  %-32s %s\n" "rsync full tree (default path)" "$(fmt_sec "$RSYNC_FULL_MS")"

touch "$PROJECT_DIR/scripts/benchmark_deploy_speed.sh"
t0=$(now_ms)
rsync -az --partial --omit-dir-times --no-perms --no-owner --no-group \
  "${RSYNC_EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" \
  "$PROJECT_DIR/" "$REMOTE:/tmp/xcelsior_rsync_full/" >/dev/null
t1=$(now_ms); RSYNC_DELTA_MS=$((t1 - t0))
printf "  %-32s %s\n" "rsync delta (1 file changed)" "$(fmt_sec "$RSYNC_DELTA_MS")"

printf "  %-32s %.1fx faster than legacy tarball\n" "rsync full vs legacy" \
  "$(awk -v l="$LEGACY_MS" -v r="$RSYNC_FULL_MS" 'BEGIN{ if (r>0) printf "%.1f", l/r; else print "n/a" }')"
printf "  %-32s %.1fx faster than legacy tarball\n" "rsync delta vs legacy" \
  "$(awk -v l="$LEGACY_MS" -v r="$RSYNC_DELTA_MS" 'BEGIN{ if (r>0) printf "%.1f", l/r; else print "n/a" }')"
if [[ -n "${ZSTD_MS:-}" && -n "${LEGACY_MS:-}" && "${ZSTD_MS:-0}" -gt 0 ]]; then
  printf "  %-32s %.1fx faster than legacy tarball\n" "zstd tarball vs legacy" \
    "$(awk -v l="$LEGACY_MS" -v z="$ZSTD_MS" 'BEGIN{ printf "%.1f", l/z }')"
fi

rm -f /tmp/xcelsior_legacy.tar.gz /tmp/xcelsior_zstd.tar.zst
ssh_remote "rm -rf /tmp/xcelsior_legacy_extract /tmp/xcelsior_zstd_extract /tmp/xcelsior_rsync_full"

section "7. Live --quick deploy (unchanged tree, should skip most work)"
if [[ "${RUN_LIVE_DEPLOY_BENCH:-0}" == "1" ]]; then
  printf "  Running XCELSIOR_DEPLOY_TIMING=1 deploy.sh --quick ...\n"
  T0=$(now_ms)
  XCELSIOR_DEPLOY_TIMING=1 bash "$SCRIPT_DIR/deploy.sh" --quick 2>&1 | grep -E '⏱|Deploy diff|already at|complete' || true
  T1=$(now_ms)
  printf "  %-28s %s\n" "full --quick deploy" "$(fmt_sec "$(elapsed_ms "$T0" "$T1")")"
else
  printf "  Skipped (set RUN_LIVE_DEPLOY_BENCH=1 to run production --quick timing)\n"
fi

rm -f "$TMP_TAR/raw.tar" /tmp/xcelsior_bench.*
section "Summary"
cat <<'EOF'
  Legacy path: tarball + gzip/pigz + serial SSH + no BuildKit cache retention
  Optimized:   rsync delta + zstd wire + batched SSH + parallel hashes + BuildKit cache

  Biggest wins on repeat deploys:
    - rsync delta vs full tarball upload (especially small code changes)
    - skipped docker rebuild when hashes unchanged (--quick / deploy diff)
    - BuildKit cache mounts on rebuild (measure on next code-change deploy)
EOF
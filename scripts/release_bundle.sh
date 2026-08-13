#!/usr/bin/env bash
# Assemble, verify and publish an immutable model release bundle.
#
#   scripts/release_bundle.sh create runs/hydranet_indoor v1
#   scripts/release_bundle.sh verify releases/v1
#   scripts/release_bundle.sh publish releases/v1 gs://syncai-hydranet
#
# A model version is not a checkpoint file. It is the weights, the graph, the config,
# the lineage and the numbers, frozen together and checksummed. Git branches version the
# code; nothing in git versions the model, because runs/ is gitignored.
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

sha() { sha256sum "$1" | cut -d' ' -f1; }

# ---------------------------------------------------------------- create

create() {
  local run_dir="${1:?usage: create <run_dir> <version>}"
  local version="${2:?usage: create <run_dir> <version>}"
  local out="releases/$version"

  [ -d "$run_dir" ] || die "no such run directory: $run_dir"
  [ -e "$out" ] && die "$out already exists; a release is immutable, pick a new version"
  for f in best.pt config.yaml meta.json metrics.jsonl; do
    [ -f "$run_dir/$f" ] || die "$run_dir/$f is missing; not a complete run"
  done

  # Gate 1: the code that produced these weights must exist in git. A dirty tree means
  # the exact source is only in uncommitted.patch, which is a recovery route, not a
  # release. Cutting a release from one publishes weights nobody can reproduce.
  local commit dirty
  commit=$(python3 -c "import json;print(json.load(open('$run_dir/meta.json'))['git']['commit'])")
  dirty=$(python3 -c "import json;print(json.load(open('$run_dir/meta.json'))['git']['dirty'])")
  [ "$dirty" = "False" ] || die "run was trained from a dirty tree; commit the code and retrain"
  git cat-file -e "$commit^{commit}" 2>/dev/null || die "commit $commit is not in this repository"
  git merge-base --is-ancestor "$commit" HEAD 2>/dev/null \
    || echo "warning: $commit is not an ancestor of HEAD; releasing off-branch work"

  mkdir -p "$out"
  cp "$run_dir/best.pt" "$out/model.pt"
  cp "$run_dir/config.yaml" "$out/config.yaml"
  cp "$run_dir/meta.json" "$out/meta.json"

  # Gate 2: the exported graph must agree with the checkpoint. Without this, model.pt
  # and model.onnx sitting in one directory is a claim, not a fact.
  echo "exporting ONNX with the parity gate..."
  ${PYTHON:-python3} -m syncai_hydranet.cli.export_onnx \
    --config "$out/config.yaml" --checkpoint "$out/model.pt" \
    --output "$out/model.onnx" --check-parity >"$out/export.log" 2>&1 \
    || { tail -20 "$out/export.log"; die "export or parity check failed"; }
  grep -q "PASS" "$out/export.log" || die "parity did not pass; see $out/export.log"

  # The best epoch by the run's own primary metric, which is the checkpoint being shipped.
  python3 - "$run_dir" "$out" <<'PY'
import json, sys
run, out = sys.argv[1], sys.argv[2]
rows = [json.loads(x) for x in open(f"{run}/metrics.jsonl") if x.strip()]
key = rows[-1].get("primary_metric")
best = max((r for r in rows if key in r), key=lambda r: r[key])
json.dump({"primary_metric": key, "selected_epoch": best.get("epoch"),
           "epochs_run": rows[-1].get("epoch"), "metrics": best},
          open(f"{out}/metrics.json", "w"), indent=2)
print(f"  selected epoch {best.get('epoch')} on {key} = {best[key]:.4f}")
PY

  # The manifest is what makes the bundle verifiable. Object versioning protects against
  # deletion; only checksums answer "has any of this been altered".
  python3 - "$out" "$version" "$commit" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
out, version, commit = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
files = {}
for p in sorted(out.iterdir()):
    if p.name in ("MANIFEST.json", "export.log") or not p.is_file():
        continue
    files[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
created = subprocess.run(["git", "log", "-1", "--format=%cI", commit],
                         capture_output=True, text=True).stdout.strip()
json.dump({"version": version, "source_commit": commit, "commit_date": created,
           "files": files}, open(out / "MANIFEST.json", "w"), indent=2)
print(f"  manifest: {len(files)} files")
PY

  echo
  echo "created $out"
  ls -lh "$out" | awk 'NR>1 {print "  " $5, $9}'
  echo
  echo "Engines are NOT in this bundle. A TensorRT engine is tied to a GPU architecture,"
  echo "a TensorRT version and a JetPack version, so it is a per-target build artefact."
  echo "Build it on the board and drop it in $out/engines/<board>_<jetpack>_<trt>_<prec>.engine"
}

# ---------------------------------------------------------------- verify

verify() {
  local dir="${1:?usage: verify <bundle_dir>}"
  [ -f "$dir/MANIFEST.json" ] || die "$dir has no MANIFEST.json"
  local bad=0
  while IFS=$'\t' read -r name want; do
    got=$(sha "$dir/$name" 2>/dev/null || echo MISSING)
    if [ "$got" = "$want" ]; then
      printf "  ok    %s\n" "$name"
    else
      printf "  FAIL  %s\n" "$name"; bad=1
    fi
  done < <(python3 -c "
import json,sys
m=json.load(open('$dir/MANIFEST.json'))
for k,v in m['files'].items(): print(k+chr(9)+v['sha256'])
")
  [ "$bad" = 0 ] || die "bundle does not match its manifest"
  echo "bundle verified"
}

# ---------------------------------------------------------------- publish

publish() {
  local dir="${1:?usage: publish <bundle_dir> <gs://bucket>}"
  local bucket="${2:?usage: publish <bundle_dir> <gs://bucket>}"
  local version
  version=$(python3 -c "import json;print(json.load(open('$dir/MANIFEST.json'))['version'])")
  verify "$dir"
  local dest="$bucket/releases/$version"
  gcloud storage ls "$dest" >/dev/null 2>&1 && die "$dest exists; releases are immutable"
  gcloud storage rsync -r "$dir" "$dest"
  echo "published $dest"
}

case "${1:-}" in
  create)  shift; create "$@" ;;
  verify)  shift; verify "$@" ;;
  publish) shift; publish "$@" ;;
  *) die "usage: $0 {create <run_dir> <version>|verify <dir>|publish <dir> <gs://bucket>}" ;;
esac

#!/usr/bin/env bash
#
# run_benchmark.sh -- run one benchmark config end to end and record what it cost.
#
# Runs the four parakeet stages separately rather than through `parakeet run`,
# so each one can be timed on its own: the multislice (exit_wave) dominates,
# and lumping it in with the rest hides that.
#
# Usage:
#   benchmark/run_benchmark.sh spa_70s_small
#   benchmark/run_benchmark.sh tilt_series_70s --outdir /scratch/$USER/bench
#   benchmark/run_benchmark.sh spa_70s --gpu 0 --dtype float32
#
# Writes <outdir>/<config>/result.json alongside the simulation output.
# Summarise several runs with: benchmark/summarise.py <outdir>
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="${1:-}"
shift || true

if [ -z "$CONFIG_NAME" ]; then
    echo "usage: $(basename "$0") <config-name> [--outdir DIR] [--gpu N] [--dtype T] [--keep-intermediate]" >&2
    echo "available:" >&2
    for f in "$HERE"/configs/*.yaml; do echo "  $(basename "$f" .yaml)" >&2; done
    exit 2
fi

CONFIG="$HERE/configs/${CONFIG_NAME%.yaml}.yaml"
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 2; }

OUTDIR="${TMPDIR:-/tmp}/parakeet_benchmark"
GPU=0
DTYPE=""
KEEP=0
while [ $# -gt 0 ]; do
    case "$1" in
        --outdir) OUTDIR="$2"; shift 2 ;;
        --gpu)    GPU="$2"; shift 2 ;;
        --dtype)  DTYPE="$2"; shift 2 ;;
        --keep-intermediate) KEEP=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

RUN="$OUTDIR/$(basename "$CONFIG" .yaml)"
mkdir -p "$RUN"
SAMPLE="$RUN/sample.h5"
EXIT_WAVE="$RUN/exit_wave.h5"
OPTICS="$RUN/optics.h5"
IMAGE="$RUN/image.mrc"
RESULT="$RUN/result.json"
LOG="$RUN/run.log"

DTYPE_ARG=()
[ -n "$DTYPE" ] && DTYPE_ARG=(--dtype "$DTYPE")

echo "config : $CONFIG"
echo "output : $RUN"
echo "gpu    : $GPU"
: > "$LOG"

# stage -> seconds, collected as we go
declare -A TIMING

time_stage () {
    local label="$1"; shift
    echo "--- $label" | tee -a "$LOG"
    local t0 t1
    t0=$(date +%s.%N)
    if ! "$@" >>"$LOG" 2>&1; then
        echo "FAILED at $label -- see $LOG" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    t1=$(date +%s.%N)
    TIMING[$label]=$(echo "$t1 - $t0" | bc)
    printf '    %.1f s\n' "${TIMING[$label]}"
}

time_stage sample \
    parakeet sample new -c "$CONFIG" -s "$SAMPLE"
time_stage add_molecules \
    parakeet sample add_molecules -c "$CONFIG" -s "$SAMPLE"
time_stage add_fiducials \
    parakeet sample add_fiducials -c "$CONFIG" -s "$SAMPLE"
time_stage exit_wave \
    parakeet simulate exit_wave -c "$CONFIG" -s "$SAMPLE" -e "$EXIT_WAVE" -d gpu --gpu_id "$GPU"
time_stage optics \
    parakeet simulate optics -c "$CONFIG" -e "$EXIT_WAVE" -o "$OPTICS" -d gpu --gpu_id "$GPU" "${DTYPE_ARG[@]}"
time_stage image \
    parakeet simulate image -c "$CONFIG" -o "$OPTICS" -i "$IMAGE" "${DTYPE_ARG[@]}"

# Record the run. Everything quoted anywhere should come from here, not from
# a number typed into a document.
python3 - "$CONFIG" "$RESULT" "$IMAGE" "$SAMPLE" "$GPU" "${!TIMING[@]}" -- "${TIMING[@]}" <<'PY'
import json, os, subprocess, sys, datetime

config, result, image, sample, gpu = sys.argv[1:6]
rest = sys.argv[6:]
split = rest.index("--")
labels, values = rest[:split], rest[split + 1:]
timing = {k: round(float(v), 2) for k, v in zip(labels, values)}

def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader", "-i", str(gpu)],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:
        return None

import parakeet.config
c = parakeet.config.load(config)
order = ["sample", "add_molecules", "add_fiducials", "exit_wave", "optics", "image"]

json.dump({
    "config": os.path.basename(config),
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "gpu": gpu_name(),
    "parakeet_version": getattr(__import__("parakeet"), "__version__", None),
    "specimen": {
        "detector_px": [c.microscope.detector.nx, c.microscope.detector.ny],
        "pixel_size_A": c.microscope.detector.pixel_size,
        "ice_thickness_A": c.sample.box[2],
        "ribosomes": c.sample.molecules.pdb[0].instances,
        "fiducials": c.sample.fiducials.count if c.sample.fiducials else 0,
        "scan_mode": str(c.scan.mode),
        "num_images": c.scan.num_images,
        "dose_per_image": c.microscope.beam.electrons_per_angstrom,
    },
    "seconds": {k: timing[k] for k in order if k in timing},
    "seconds_total": round(sum(timing.values()), 2),
    "output_bytes": {
        "sample": os.path.getsize(sample) if os.path.exists(sample) else None,
        "image": os.path.getsize(image) if os.path.exists(image) else None,
    },
}, open(result, "w"), indent=1)
print(f"\nrecorded -> {result}")
PY

if [ "$KEEP" -eq 0 ]; then
    rm -f "$EXIT_WAVE" "$OPTICS"
    echo "removed intermediates (--keep-intermediate to retain them)"
fi

echo "image  : $IMAGE"

#!/usr/bin/env bash
# Portable version of run_all.sh (albertop27 fork, branch `local`): no hardcoded author paths, no cloud.
# Feed it any generated laptop GLB (local TRELLIS 2 / Pixal3D, or a cached Tripo one).
#
#   bash laptop/run_local.sh <model.glb> [gen=trellis2] [out_dir=laptop/out_local]
#
# Env: BLENDER (default: Blender 5.3 install path), FFMPEG (default: ffmpeg on PATH)
set -e
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && (pwd -W 2>/dev/null || pwd))"
GLB="$1"; GEN="${2:-trellis2}"; OUT="${3:-$D/out_local}"
[ -z "$GLB" ] && { echo "usage: run_local.sh <model.glb> [gen] [out_dir]"; exit 1; }
BLENDER="${BLENDER:-/c/Program Files/Blender Foundation/Blender 5.3/blender.exe}"
FFMPEG="${FFMPEG:-ffmpeg}"
G="$D/scripts/blender_gpu.py"
mkdir -p "$OUT"
B() { "$BLENDER" -b --factory-startup "$@" 2>&1 | grep -E "BUILT|CHECK|EXPORTED|RENDERED|Error|Traceback" || true; }
B --python "$D/scripts/build.py" -- --glb "$GLB" --gen "$GEN" --ui "$D/refs/screen_ui_0.png" --out "$OUT/laptop.blend"
B --python "$D/scripts/check.py" -- --blend "$OUT/laptop.blend" --out "$OUT/check.json" --step 1
B --python "$D/scripts/export.py" -- --blend "$OUT/laptop.blend" --out "$OUT/laptop.glb"
rm -rf "$OUT/stills" "$OUT/anim"
B --python "$G" --python "$D/scripts/render.py" -- --blend "$OUT/laptop.blend" --out "$OUT/stills" --stills 0,90,135 --samples 64 --resx 1280 --resy 720 --cam persp
B --python "$G" --python "$D/scripts/render.py" -- --blend "$OUT/laptop.blend" --out "$OUT/stills" --stills 0,45,90,135 --samples 48 --resx 960 --resy 720 --cam hinge
B --python "$G" --python "$D/scripts/render.py" -- --blend "$OUT/laptop.blend" --out "$OUT/anim" --anim --samples 32 --resx 960 --resy 540 --cam persp
cd "$OUT"
"$FFMPEG" -y -loglevel error -framerate 24 -i anim/f_%04d.png -c:v libx264 -preset slow -crf 18 -movflags +faststart -pix_fmt yuv420p laptop_hinge.mp4
"$FFMPEG" -y -loglevel error -framerate 24 -i anim/f_%04d.png -vf "fps=16,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" laptop_hinge.gif
echo DONE

#!/bin/bash
# spirv.sh - compile all .comp shaders to SPIR-V ASM + analysis
# Usage:
#   ./spirv.sh                 # default: full pipeline, single-threaded
#   ./spirv.sh --only-asm      # compile & disassemble only
#   ./spirv.sh -j 4            # use 4 concurrent workers
#   ./spirv.sh --no-cross --no-cfg --no-reflect --no-debug

set -euo pipefail

SRC_DIR="${SRC_DIR:-$HOME/Documents/git/qoa/src/kernel/shaders}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Documents/git/qoa/verification/spirv}"

# Defaults (can be changed via CLI flags)
ONLY_ASM=0
DO_CROSS=1
DO_CFG=1
DO_REFLECT=1
DO_DEBUG=1
JOBS=1

print_usage() {
    cat <<EOF
Usage: $0 [options]
Options:
  --only-asm        Only produce .spvasm (no cross-compile/CFG/reflect/debug)
  --no-cross        Skip spirv-cross (HLSL / MSL)
  --no-cfg          Skip CFG generation
  --no-reflect      Skip spirv-reflect
  --no-debug        Skip debug build and debug assembly
  -j N              Run up to N shaders in parallel (default 1)
  -h, --help        Show this help
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only-asm) ONLY_ASM=1; DO_CROSS=0; DO_CFG=0; DO_REFLECT=0; DO_DEBUG=0; shift ;;
        --no-cross) DO_CROSS=0; shift ;;
        --no-cfg) DO_CFG=0; shift ;;
        --no-reflect) DO_REFLECT=0; shift ;;
        --no-debug) DO_DEBUG=0; shift ;;
        -j) JOBS="$2"; shift 2 ;;
        --help|-h) print_usage; exit 0 ;;
        *) echo "Unknown arg: $1"; print_usage; exit 1 ;;
    esac
done

check_dependencies() {
    local missing=()
    # Required for minimal flow
    for cmd in glslc spirv-dis spirv-opt spirv-val; do
        if ! command -v "$cmd" &>/dev/null; then missing+=("$cmd"); fi
    done
    # Optional but recommended
    if [ "$DO_CFG" -eq 1 ]; then
        if ! command -v spirv-cfg &>/dev/null; then missing+=("spirv-cfg"); fi
    fi
    if [ "$DO_REFLECT" -eq 1 ]; then
        if ! command -v spirv-reflect &>/dev/null; then missing+=("spirv-reflect"); fi
    fi
    if [ "$DO_CROSS" -eq 1 ]; then
        if ! command -v spirv-cross &>/dev/null; then missing+=("spirv-cross"); fi
    fi
    if ! command -v spirv-opt &>/dev/null; then missing+=("spirv-opt"); fi
    if ! command -v dot &>/dev/null && [ "$DO_CFG" -eq 1 ]; then
        # dot optional; let user know but not fatal
        echo "Warning: Graphviz 'dot' not found; CFG PNGs won't be generated."
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        echo "Error: missing commands: ${missing[*]}"
        echo "Install the Vulkan SDK or required packages and try again."
        exit 1
    fi
}

# process a single shader (isolated)
process_shader() {
    local shader="$1"
    local base
    base=$(basename "$shader" .comp)
    local tmp_bin tmp_opt tmp_dbg
    tmp_bin=$(mktemp)
    tmp_opt=$(mktemp)
    tmp_dbg=$(mktemp)

    local asm="$OUTPUT_DIR/${base}.spvasm"
    local opt_asm="$OUTPUT_DIR/${base}_opt.spvasm"
    local dbg_asm="$OUTPUT_DIR/${base}_debug.spvasm"
    local cfg_dot="$OUTPUT_DIR/${base}_cfg.dot"
    local cfg_png="$OUTPUT_DIR/${base}_cfg.png"
    local hlsl="$OUTPUT_DIR/${base}.hlsl"
    local metal="$OUTPUT_DIR/${base}.metal"
    local reflect_txt="$OUTPUT_DIR/${base}_reflect.txt"
    local stats_txt="$OUTPUT_DIR/${base}_stats.txt"

    echo -e "\n=== Processing: $base ==="

    # compile release
    if ! glslc "$shader" -o "$tmp_bin" -O; then
        echo "ERROR: glslc failed for $base"; rm -f "$tmp_bin" "$tmp_opt" "$tmp_dbg"; return 1
    fi

    # disassemble -> asm
    spirv-dis "$tmp_bin" -o "$asm"

    # validate
    spirv-val "$tmp_bin" >/dev/null || echo "Warning: spirv-val failed for $base"

    # optimize + disasm
    spirv-opt "$tmp_bin" -O -o "$tmp_opt"
    spirv-dis "$tmp_opt" -o "$opt_asm"

    # CFG
    if [ "$DO_CFG" -eq 1 ]; then
        if spirv-cfg "$tmp_bin" > "$cfg_dot" 2>/dev/null; then
            if command -v dot &>/dev/null; then
                dot -Tpng "$cfg_dot" -o "$cfg_png" || echo "dot failed for $cfg_dot"
            fi
        else
            echo "Warning: spirv-cfg failed for $base"
            rm -f "$cfg_dot" || true
        fi
    fi

    # cross-compile
    if [ "$DO_CROSS" -eq 1 ]; then
        if spirv-cross "$tmp_bin" --hlsl > "$hlsl" 2>/dev/null; then
            echo "HLSL -> $hlsl"
        else
            echo "Warning: spirv-cross --hlsl failed for $base"; rm -f "$hlsl" || true
        fi
        if spirv-cross "$tmp_bin" --msl > "$metal" 2>/dev/null; then
            echo "MSL -> $metal"
        else
            echo "Warning: spirv-cross --msl failed for $base"; rm -f "$metal" || true
        fi
    fi

    # reflection
    if [ "$DO_REFLECT" -eq 1 ]; then
        if spirv-reflect "$tmp_bin" > "$reflect_txt" 2>/dev/null; then
            echo "Reflect -> $reflect_txt"
        else
            echo "Warning: spirv-reflect failed for $base"; rm -f "$reflect_txt" || true
        fi
    fi

    # debug build
    if [ "$DO_DEBUG" -eq 1 ]; then
        if glslc "$shader" -o "$tmp_dbg" -g -O0; then
            spirv-dis "$tmp_dbg" -o "$dbg_asm"
        else
            echo "Warning: debug glslc failed for $base"; rm -f "$dbg_asm" || true
        fi
    fi

    # stats (instruction count and sizes)
    {
        echo "Instruction count: $(grep -E '^%[0-9]+' "$asm" | wc -l)"
        echo
        echo "Original binary size:"
        ls -lh "$tmp_bin" || true
        echo
        echo "Optimized binary size:"
        ls -lh "$tmp_opt" || true
    } > "$stats_txt"

    echo "Produced: $asm ${opt_asm:+$opt_asm} ${dbg_asm:+$dbg_asm} $stats_txt"

    # cleanup temps
    rm -f "$tmp_bin" "$tmp_opt" "$tmp_dbg"
    return 0
}

main() {
    check_dependencies
    mkdir -p "$OUTPUT_DIR"

    shopt -s nullglob
    local shaders=( "$SRC_DIR"/*.comp )
    if [ ${#shaders[@]} -eq 0 ]; then
        echo "No .comp shaders found in $SRC_DIR"
        exit 0
    fi

    # concurrency launcher
    local launched=0
    for s in "${shaders[@]}"; do
        # respect JOBS
        while [ "$JOBS" -gt 1 ] && [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
            sleep 0.10
        done

        # run in background or foreground depending on JOBS
        if [ "$JOBS" -gt 1 ]; then
            ( process_shader "$s" ) &   # subshell; failure won't kill whole script
        else
            ( process_shader "$s" ) || echo "shader failed: $s"
        fi
        launched=$((launched+1))
    done

    # wait for background jobs if any
    if [ "$JOBS" -gt 1 ]; then
        wait
    fi

    echo -e "\nAll done. Results: $OUTPUT_DIR"
}

main

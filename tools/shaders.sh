#!/bin/sh
# fitxategi honek GPUaren kernela konpilatzen du: src/shaders/tile.comp -> src/shaders/tile.spv (SPIR-V,
# vulkan 1.2), egiaztatzen du (spirv-val), eta iturburuaren marka idazten du tile.spv.fnv-n (FNV-1a 64);
# tests/gpu.rs-ek marka horrekin ziurtatzen du .spv iturburu berritik datorrela.
#
# behar ditu: glslangValidator eta spirv-val (debian/ubuntu: glslang-tools, spirv-tools), python3.
# konpilatutako .spv biltegian dago, QOA konpilatzeko ez da hauetako bat ere behar.
set -eu
cd "$(dirname "$0")/../src/shaders"
glslangValidator -V --target-env vulkan1.2 -o tile.spv tile.comp
spirv-val --target-env vulkan1.2 tile.spv
python3 - <<'PY'
h = 0xcbf29ce484222325
for b in open("tile.comp", "rb").read():
    h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
open("tile.spv.fnv", "w").write(f"{h:016x}\n")
PY
echo "tile.spv: $(wc -c < tile.spv) byte, marka $(cat tile.spv.fnv)"

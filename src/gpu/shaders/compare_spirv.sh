#!/bin/bash

if [[ ! -f hadamard.spv ]]; then
  echo "hadamard.spv not found!"
  exit 1
fi

if [[ ! -f dumped.spv ]]; then
  echo "dumped.spv not found! Dump your loaded SPIR-V buffer first."
  exit 1
fi

xxd hadamard.spv > hadamard_original.hex

xxd dumped.spv > hadamard_dumped.hex

diff hadamard_original.hex hadamard_dumped.hex

echo "Byte differences (if any):"

cmp -l hadamard.spv dumped.spv || echo "No differences found."

# rm hadamard_original.hex hadamard_dumped.hex

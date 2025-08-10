#!/bin/bash

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <original_shader.spv> <dumped_shader.spv>"
  exit 1
fi

original="$1"
dumped="$2"

if [[ ! -f "$original" ]]; then
  echo "error: '$original' not found!"
  exit 1
fi

if [[ ! -f "$dumped" ]]; then
  echo "error: '$dumped' not found! dump your loaded SPIR-V buffer first."
  exit 1
fi

xxd "$original" > original.hex
xxd "$dumped" > dumped.hex

diff original.hex dumped.hex

echo "byte differences (if any):"
cmp -l "$original" "$dumped" || echo "no differences found."

# cleanup
# rm original.hex dumped.hex

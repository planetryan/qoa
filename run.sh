#!/bin/sh
# fitxategi honek qoa exekutatzen du linux, macos eta antzekoetan: `./run.sh run examples/bell.qoa`.
#
#   - iturburu-kopia batean (Cargo.toml eta cargo badaude): `cargo build --release` (eguneratuta badago,
#     berehala) eta target/release/qoa
#   - argitalpen-pakete batean: bin/<helburua>/qoa, makina honen sistema eta arkitekturaren arabera
#     (x86_64, aarch64, riscv64gc, powerpc64le; linux edo macos), edo script-aren ondoko qoa. android-en
#     (termux) eta glibc gabeko linux-etan, aarch64 estatikoa (musl)
# argumentu guztiak qoa-ri pasatzen zaizkio bere horretan.
set -eu
here=$(cd "$(dirname "$0")" && pwd)

case "${QOA_LANG:-${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}}" in
    eu*) eu=1 ;;
    *) eu= ;;
esac

if [ -f "$here/Cargo.toml" ] && command -v cargo >/dev/null 2>&1; then
    cargo build --release --quiet --manifest-path "$here/Cargo.toml"
    exec "$here/target/release/qoa" "$@"
fi

os=$(uname -s)
arch=$(uname -m)
case "$os" in
    Linux) sys=unknown-linux-gnu ;;
    Darwin) sys=apple-darwin ;;
    *) sys=unknown ;;
esac
case "$arch" in
    x86_64 | amd64) cpu=x86_64 ;;
    aarch64 | arm64) cpu=aarch64 ;;
    riscv64) cpu=riscv64gc ;;
    ppc64le) cpu=powerpc64le ;;
    *) cpu=$arch ;;
esac
target="$cpu-$sys"
# android-ek ez du glibc-rik: bitar estatikoa bakarrik
if [ -n "${ANDROID_ROOT:-}" ]; then
    target="$cpu-unknown-linux-musl"
fi

for exe in "$here/bin/$target/qoa" "$here/bin/$cpu-unknown-linux-musl/qoa" "$here/qoa"; do
    if [ -x "$exe" ]; then
        exec "$exe" "$@"
    fi
done

if [ -n "$eu" ]; then
    echo "run.sh: ez dago qoa bitarrik $os $arch-rako (bin/$target/qoa bilatu da)." >&2
    echo "konpilatu iturburutik: https://www.rust-lang.org/tools/install eta gero ./run.sh berriz" >&2
else
    echo "run.sh: no qoa binary for $os $arch (looked for bin/$target/qoa)." >&2
    echo "build it from source: install Rust (https://www.rust-lang.org/tools/install) and run ./run.sh again in a source checkout" >&2
fi
exit 1

# Contributing to QOA

*[Euskaraz: CONTRIBUTING.eu.md](CONTRIBUTING.eu.md)*

Thank you for helping. This page is how QOA is built, checked and written, so that a change fits in.

## Build and check

```
cargo build --release
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

CI runs the same on Linux, Windows and macOS, and the core suites under qemu on ARM64, RISC-V (vector
lengths 128 and 512), IBM POWER10, IBM Z (big-endian) and static ARM64. A change is ready when all of it is
green.

## Rules the code keeps

- **Nothing pretends.** An instruction or option that is not implemented is an error that says so, never a
  silent no-op or an approximation presented as exact. Results say how they were computed.
- **Accuracy before speed.** The physics runs in 64-bit floating point or better; sums over the state use
  double-double arithmetic (`src/dd.rs`). Lower-precision hardware may draw pictures, never compute results.
- **Two languages.** Every message, error and help text exists in English and Basque, through the `t!` and
  `ts!` macros (`src/lang.rs`); `QOA_LANG=eu` or `QOA_LANG=en` picks one.
- **Comments in Basque**, lowercase, with `//` (no `///` or `//!`). `tests/style.rs` checks this for every
  source file in the repository.
- **Every claim is tested.** An example's comments say what it should print, and `tests/programs.rs` checks
  it; numbers in the docs come from a test or a script in `tools/`.

## Writing docs and release notes

Docs, release notes and commit messages are for anyone who reads them, now or years later:

- Explain what changed and why in terms a reader can check: the command, the result, the test that proves it.
- Write for someone who was not part of any discussion: no references to private conversations, personal to-do
  lists or who decided what. Open design points are phrased as open points, not as questions to one person.
- Use versions and dates, not "now", "recently" or "new".
- Keep English and Basque in step: `name.md` and `name.eu.md`; release notes are `changelog/QOA vX.Y.Z.md`,
  English first and Basque below.
- Say plainly what is not done or not supported.

## Releasing

1. Set the version in `Cargo.toml` and write `changelog/QOA vX.Y.Z.md`.
2. Wait for green CI on the commit, then bring `main` to it.
3. Run the `release` workflow on `main` with `release` set to `vX.Y.Z` (from the Actions tab or the GitHub
   app). It builds every system, tests the bundle on Linux, Windows and macOS, tags the commit and publishes
   the release with the changelog as its notes.

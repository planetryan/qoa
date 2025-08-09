#!/usr/bin/env bash
# install_vulkan_sdk.sh
# Auto-detects QOA root dir from this script location,
# installs latest Vulkan SDK into $QOA_ROOT/verification/sdk,
# and cleans up older SDK versions.

set -euo pipefail

# Resolve the absolute directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Assume QOA root is 3 levels up from script dir (adjust if needed)
QOA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Installation directory inside QOA root
INSTALL_PARENT="$QOA_ROOT/verification/sdk"

TMPDIR="${TMPDIR:-/tmp}"
KEEP_COUNT="${KEEP_COUNT:-1}"   # Keep N newest versions
FORCE="${FORCE:-0}"             # 1 to skip prompts (not used currently)

SDK_BASE_URL="https://sdk.lunarg.com/sdk"

usage() {
    cat <<EOF
Usage: $0 [--install-parent DIR] [--keep N] [--yes]

Options:
  --install-parent DIR   Install SDKs under DIR (default: $INSTALL_PARENT)
  --keep N               Keep N newest versioned SDK directories (default: $KEEP_COUNT)
  --yes                  Run non-interactively (no prompts). Default: auto-delete without prompt.
  -h, --help             Show this help.
EOF
}

# Argument parsing (override INSTALL_PARENT and KEEP_COUNT)
while (( "$#" )); do
    case "$1" in
        --install-parent) INSTALL_PARENT="$2"; shift 2 ;;
        --keep) KEEP_COUNT="$2"; shift 2 ;;
        --yes|--y) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "$INSTALL_PARENT"
cd "$INSTALL_PARENT"

echo "QOA root detected as: $QOA_ROOT"
echo "Installing Vulkan SDK into: $INSTALL_PARENT"
echo "Querying latest SDK version..."

set +o pipefail
LATEST_VERSION=$(curl -fsSL "https://vulkan.lunarg.com/sdk/latest/linux.txt" || true)
set -o pipefail

if [ -z "$LATEST_VERSION" ]; then
    echo "Failed to get latest version from LunarG. Trying generic 'latest' download URL..."
    DOWNLOAD_URL="${SDK_BASE_URL}/download/latest/linux/vulkan_sdk.tar.xz"
    TARFILE="$TMPDIR/vulkan_sdk-latest.tar.xz"
    echo "Downloading: $DOWNLOAD_URL"
    curl -fSL -o "$TARFILE" "$DOWNLOAD_URL"
    echo "Extracting archive..."
    tar -xvf "$TARFILE" -C "$INSTALL_PARENT"
    rm -f "$TARFILE"
    echo "Done. Note: could not determine explicit version string (used 'latest')."
    exit 0
fi

echo "Latest version: $LATEST_VERSION"

FILE_NAME="vulkansdk-linux-x86_64-${LATEST_VERSION}.tar.xz"
DOWNLOAD_URL="${SDK_BASE_URL}/download/${LATEST_VERSION}/linux/${FILE_NAME}"
TARFILE="$TMPDIR/${FILE_NAME}"

echo "Downloading: $DOWNLOAD_URL"
curl -fSL -o "$TARFILE" "$DOWNLOAD_URL" || {
    echo "Download failed. If you hit a rate limit, try again later or append '?Human=true' to the URL."
    exit 1
}

echo "Verifying SHA (if available)..."
SHA_URL="${SDK_BASE_URL}/sdk/sha/${LATEST_VERSION}/linux/${FILE_NAME}.txt"
if curl -fsSL "$SHA_URL" -o /dev/null 2>/dev/null; then
    sha_expected=$(curl -fsSL "$SHA_URL" | awk '{print $1}')
    sha_actual=$(sha256sum "$TARFILE" | awk '{print $1}')
    if [ "$sha_expected" = "$sha_actual" ]; then
        echo "SHA256 OK"
    else
        echo "Warning: SHA256 mismatch! Expected: $sha_expected, Actual: $sha_actual"
        echo "Aborting."
        rm -f "$TARFILE"
        exit 1
    fi
else
    echo "No SHA available from server; skipping checksum verification."
fi

echo "Extracting $TARFILE into $INSTALL_PARENT..."
tar -xvf "$TARFILE" -C "$INSTALL_PARENT" || { echo "Extraction failed"; rm -f "$TARFILE"; exit 1; }

EXTRACTED_DIR="$INSTALL_PARENT/$LATEST_VERSION"
if [ ! -d "$EXTRACTED_DIR" ]; then
    echo "Expected extracted dir $EXTRACTED_DIR not found. Listing newly created dirs:"
    ls -1dt "$INSTALL_PARENT"/*/ | head -10
fi

rm -f "$TARFILE"
echo "Extraction done. Cleaned up tarball."

echo "Removing older versioned SDK directories (keeping $KEEP_COUNT newest)..."

mapfile -t sdk_dirs < <(find "$INSTALL_PARENT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+' | sort -Vr)

if [ "${#sdk_dirs[@]}" -le "$KEEP_COUNT" ]; then
    echo "Found ${#sdk_dirs[@]} versioned SDK dirs; nothing to delete."
else
    to_delete=( "${sdk_dirs[@]:$KEEP_COUNT}" )
    printf 'Will remove %d old SDK dirs:\n' "${#to_delete[@]}"
    for d in "${to_delete[@]}"; do printf '  %s\n' "$d"; done

    for d in "${to_delete[@]}"; do
        full="$INSTALL_PARENT/$d"
        if [[ "$full" == "$INSTALL_PARENT/"* ]] && [[ "$d" =~ ^[0-9] ]]; then
            echo "Removing $full"
            rm -rf -- "$full"
        else
            echo "Skipping suspicious path: $full"
        fi
    done
fi

echo "Done. Current SDKs in $INSTALL_PARENT:"
ls -1d "$INSTALL_PARENT"/*/ 2>/dev/null || true

echo "Add the new SDK env setup to your shell rc:"
echo "  source $EXTRACTED_DIR/setup-env.sh"

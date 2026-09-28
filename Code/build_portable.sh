#!/usr/bin/env bash
# build_portable.sh — rebuild dist/portable/ from scratch with one command.
#
# Usage:
#   cd Code/Python/EnergyDemandForecaster
#   bash build_portable.sh
#
# What it does:
#   1. Downloads python-build-standalone 3.12 for Mac arm64 + Windows x86_64
#      (caches downloads in dist/downloads/ — skips if already present)
#   2. Wipes and recreates dist/portable/{mac,windows}/
#   3. Installs packages into each platform's packages/ dir
#   4. Copies app/, src/, and the PJME sample CSV into both platforms
#   5. Writes the launcher scripts
#   6. Reports bundle sizes
#
# Requirements on this Mac:
#   curl, tar, unzip (all ship with macOS)
#   Internet connection (first run only — subsequent runs use the cache)

set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIST="$SCRIPT_DIR/dist"
DOWNLOADS="$DIST/downloads"
PORTABLE="$DIST/portable"
MAC_OUT="$PORTABLE/mac"
WIN_OUT="$PORTABLE/windows"

# ── python-build-standalone release ────────────────────────────────────────────
# Pin to a specific release tag so the build is reproducible.
# To upgrade: change RELEASE_TAG and the two filenames below, then delete
# the cached tarballs in dist/downloads/ so they are re-downloaded.
RELEASE_TAG="20260508"
PY_VERSION="3.12.13"

MAC_TARBALL="cpython-${PY_VERSION}+${RELEASE_TAG}-aarch64-apple-darwin-install_only_stripped.tar.gz"
WIN_TARBALL="cpython-${PY_VERSION}+${RELEASE_TAG}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"

BASE_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${RELEASE_TAG}"

# ── helpers ────────────────────────────────────────────────────────────────────
log()  { echo ""; echo "▸ $*"; }
ok()   { echo "  ✓ $*"; }
size() { du -sh "$1" 2>/dev/null | cut -f1; }

# ── step 0: directories ────────────────────────────────────────────────────────
log "Creating directories"
mkdir -p "$DOWNLOADS"
mkdir -p "$MAC_OUT" "$WIN_OUT"

# ── step 1: download python-build-standalone (cached) ─────────────────────────
log "Downloading Python runtimes (cached in dist/downloads/)"

for TARBALL in "$MAC_TARBALL" "$WIN_TARBALL"; do
    DEST="$DOWNLOADS/$TARBALL"
    if [ -f "$DEST" ]; then
        ok "Already cached: $TARBALL"
    else
        echo "  Downloading $TARBALL …"
        curl -fL --retry 5 --retry-delay 3 \
            "$BASE_URL/$TARBALL" -o "$DEST"
        ok "Downloaded: $TARBALL"
    fi
done

# ── step 2: extract runtimes ──────────────────────────────────────────────────
log "Extracting Mac Python → mac/python/"
rm -rf "$MAC_OUT/python"
# The tarball top-level is 'python/' — extract and rename
TMP_MAC="$DIST/tmp_mac_py"
rm -rf "$TMP_MAC" && mkdir -p "$TMP_MAC"
tar -xzf "$DOWNLOADS/$MAC_TARBALL" -C "$TMP_MAC"
mv "$TMP_MAC/python" "$MAC_OUT/python"
rm -rf "$TMP_MAC"
ok "Mac Python $(size "$MAC_OUT/python")"

log "Extracting Windows Python → windows/python/"
rm -rf "$WIN_OUT/python"
TMP_WIN="$DIST/tmp_win_py"
rm -rf "$TMP_WIN" && mkdir -p "$TMP_WIN"
tar -xzf "$DOWNLOADS/$WIN_TARBALL" -C "$TMP_WIN"
mv "$TMP_WIN/python" "$WIN_OUT/python"
rm -rf "$TMP_WIN"
ok "Windows Python $(size "$WIN_OUT/python")"

# ── step 3: install packages ──────────────────────────────────────────────────
MAC_PY="$MAC_OUT/python/bin/python3"
REQS="$SCRIPT_DIR/requirements-runtime.txt"

# Mac — native arm64 packages
log "Installing Mac packages (native arm64) …"
rm -rf "$MAC_OUT/packages"
"$MAC_PY" -m pip install \
    --target "$MAC_OUT/packages" \
    --upgrade \
    --retries 10 \
    --timeout 120 \
    -r "$REQS"
ok "Mac packages $(size "$MAC_OUT/packages")"

# Windows — cross-platform wheel install from this Mac
# We use the bundled Mac Python as the pip runner, but tell pip to fetch
# Windows/win_amd64 wheels so the result runs on Windows x86_64.
# torch-cpu is fetched from the official PyTorch index.
log "Installing Windows packages (cross-install for win_amd64) …"
rm -rf "$WIN_OUT/packages"
"$MAC_PY" -m pip install \
    --target "$WIN_OUT/packages" \
    --platform win_amd64 \
    --python-version "3.12" \
    --implementation cp \
    --abi cp312 \
    --only-binary=:all: \
    --upgrade \
    --retries 10 \
    --timeout 120 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r "$REQS"
ok "Windows packages $(size "$WIN_OUT/packages")"

# ── step 4: copy app source ───────────────────────────────────────────────────
log "Copying app/, src/, data/ into both bundles"

for OUT in "$MAC_OUT" "$WIN_OUT"; do
    rm -rf "$OUT/app" "$OUT/src" "$OUT/data" "$OUT/models"
    cp -r "$SCRIPT_DIR/app"  "$OUT/app"
    cp -r "$SCRIPT_DIR/src"  "$OUT/src"
    mkdir -p "$OUT/data/raw" "$OUT/models"
    # Copy PJME sample if it exists; if not, print a warning (bundle still works
    # — user can upload their own CSV via the GUI).
    CSV="$SCRIPT_DIR/data/raw/PJME_hourly.csv"
    if [ -f "$CSV" ]; then
        cp "$CSV" "$OUT/data/raw/PJME_hourly.csv"
        ok "Copied PJME_hourly.csv → $(basename "$OUT")/data/raw/"
    else
        echo "  WARNING: $CSV not found — the 'Load PJME sample' button in the GUI"
        echo "           will be disabled. Place the CSV there and re-run to enable it."
    fi
    # The GUI ships only the quantile-trained checkpoint (lstm_pjme_q70.pt).
    # The original MSE-trained checkpoint (lstm_pjme.pt) and any other
    # variants are kept in the repo's models/ for the report's comparison
    # but are intentionally NOT bundled — q70 dominated both alternatives
    # and the GUI exposes a single LSTM option.
    LSTM_CKPT_Q70="$SCRIPT_DIR/models/lstm_pjme_q70.pt"
    LSTM_SCALER="$SCRIPT_DIR/models/lstm_pjme.scaler.json"
    if [ -f "$LSTM_CKPT_Q70" ] && [ -f "$LSTM_SCALER" ]; then
        cp "$LSTM_CKPT_Q70" "$OUT/models/lstm_pjme_q70.pt"
        cp "$LSTM_SCALER"   "$OUT/models/lstm_pjme.scaler.json"
        ok "Copied LSTM (q70) checkpoint + scaler → $(basename "$OUT")/models/"
    else
        echo "  WARNING: lstm_pjme_q70.pt or scaler missing — bundle will retrain"
        echo "           the LSTM on every run. Run scripts/bake_lstm_scaler.py and"
        echo "           scripts/train_lstm_quantile.py first."
    fi
done

# ── step 5: write launcher scripts ────────────────────────────────────────────
log "Writing Mac launchers"

cat > "$MAC_OUT/run.command" << 'LAUNCHER'
#!/bin/bash
# Energy Demand Forecaster — Mac launcher
# Double-click to run. Opens the GUI in your default browser.

set -e

BUNDLE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BUNDLE_DIR"

PYTHON="$BUNDLE_DIR/python/bin/python3"
PACKAGES="$BUNDLE_DIR/packages"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Cannot find Python at $PYTHON"
    echo "Did you run first_time_setup.command yet?"
    read -p "Press Enter to close..." _
    exit 1
fi

export PYTHONPATH="$PACKAGES:$PYTHONPATH"

echo "================================================================"
echo "  Energy Demand Forecaster — starting..."
echo "  The GUI will open in your default browser shortly."
echo "  Close this terminal window to stop the server."
echo "================================================================"
echo

"$PYTHON" -m streamlit run "$BUNDLE_DIR/app/streamlit_app.py" \
    --server.headless false \
    --global.developmentMode false \
    --browser.gatherUsageStats false
LAUNCHER

cat > "$MAC_OUT/first_time_setup.command" << 'SETUP'
#!/bin/bash
# Energy Demand Forecaster — Mac first-time setup
# Run this ONCE per copy of the bundle, before the first launch.
# It clears macOS quarantine flags so Gatekeeper does not block the bundled
# Python and grants execute permission on the launcher script.

set -e

BUNDLE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================"
echo "  Energy Demand Forecaster — first-time setup"
echo "  Bundle: $BUNDLE_DIR"
echo "================================================================"
echo

echo "[1/2] Clearing macOS quarantine attributes..."
xattr -dr com.apple.quarantine "$BUNDLE_DIR" 2>/dev/null || true

echo "[2/2] Granting execute permission on launchers..."
chmod +x "$BUNDLE_DIR/run.command"
chmod +x "$BUNDLE_DIR/first_time_setup.command"
chmod +x "$BUNDLE_DIR/python/bin/python3" 2>/dev/null || true

echo
echo "Setup complete. You can now double-click run.command to launch the app."
echo
read -p "Press Enter to close..." _
SETUP

chmod +x "$MAC_OUT/run.command" "$MAC_OUT/first_time_setup.command"
ok "Mac launchers written and marked executable"

log "Writing Windows launcher"

# Use printf to write Windows line endings (\r\n) for the .bat file
printf '@echo off\r\n' > "$WIN_OUT/run.bat"
printf 'REM Energy Demand Forecaster -- Windows launcher\r\n' >> "$WIN_OUT/run.bat"
printf 'REM Double-click to run. Opens the GUI in your default browser.\r\n' >> "$WIN_OUT/run.bat"
printf '\r\n' >> "$WIN_OUT/run.bat"
printf 'setlocal\r\n' >> "$WIN_OUT/run.bat"
printf '\r\n' >> "$WIN_OUT/run.bat"
printf 'set "BUNDLE_DIR=%%~dp0"\r\n' >> "$WIN_OUT/run.bat"
printf 'if "%%BUNDLE_DIR:~-1%%"=="\\" set "BUNDLE_DIR=%%BUNDLE_DIR:~0,-1%%"\r\n' >> "$WIN_OUT/run.bat"
printf '\r\n' >> "$WIN_OUT/run.bat"
printf 'set "PYTHON=%%BUNDLE_DIR%%\\python\\python.exe"\r\n' >> "$WIN_OUT/run.bat"
printf 'set "PACKAGES=%%BUNDLE_DIR%%\\packages"\r\n' >> "$WIN_OUT/run.bat"
printf '\r\n' >> "$WIN_OUT/run.bat"
printf 'if not exist "%%PYTHON%%" (\r\n' >> "$WIN_OUT/run.bat"
printf '    echo ERROR: Cannot find Python at %%PYTHON%%\r\n' >> "$WIN_OUT/run.bat"
printf '    pause\r\n' >> "$WIN_OUT/run.bat"
printf '    exit /b 1\r\n' >> "$WIN_OUT/run.bat"
printf ')\r\n' >> "$WIN_OUT/run.bat"
printf '\r\n' >> "$WIN_OUT/run.bat"
printf 'set "PYTHONPATH=%%PACKAGES%%;%%PYTHONPATH%%"\r\n' >> "$WIN_OUT/run.bat"
printf '\r\n' >> "$WIN_OUT/run.bat"
printf 'echo ================================================================\r\n' >> "$WIN_OUT/run.bat"
printf 'echo   Energy Demand Forecaster -- starting...\r\n' >> "$WIN_OUT/run.bat"
printf 'echo   The GUI will open in your default browser shortly.\r\n' >> "$WIN_OUT/run.bat"
printf 'echo   Close this window to stop the server.\r\n' >> "$WIN_OUT/run.bat"
printf 'echo ================================================================\r\n' >> "$WIN_OUT/run.bat"
printf 'echo.\r\n' >> "$WIN_OUT/run.bat"
printf '\r\n' >> "$WIN_OUT/run.bat"
printf '"%%PYTHON%%" -m streamlit run "%%BUNDLE_DIR%%\\app\\streamlit_app.py" ^\r\n' >> "$WIN_OUT/run.bat"
printf '    --server.headless false ^\r\n' >> "$WIN_OUT/run.bat"
printf '    --global.developmentMode false ^\r\n' >> "$WIN_OUT/run.bat"
printf '    --browser.gatherUsageStats false\r\n' >> "$WIN_OUT/run.bat"
printf '\r\n' >> "$WIN_OUT/run.bat"
printf 'pause\r\n' >> "$WIN_OUT/run.bat"

ok "Windows run.bat written"

# ── step 6: write README ──────────────────────────────────────────────────────
log "Writing README.txt"

cat > "$PORTABLE/README.txt" << 'README'
================================================================
  ENERGY DEMAND FORECASTER — PORTABLE GUI
  BA26-01 Term Project
================================================================

This folder contains a self-contained copy of the Energy Demand
Forecaster GUI. It does not require Python (or anything else) to be
installed on the host machine — everything is bundled.

Pick the folder for your operating system:

  mac/        →  Apple Silicon Macs (M1/M2/M3/M4) running macOS 13+
  windows/    →  Windows 10/11, 64-bit


================================================================
  Mac — first time (one-time setup, ~10 seconds)
================================================================

1. Open the "mac" folder in Finder.
2. Right-click "first_time_setup.command" → choose "Open" from the menu.
   (macOS will warn it is from an unidentified developer — this is normal
   for unsigned scripts. Click "Open" to confirm.)
3. A terminal window will run for a few seconds and close itself.

You only need to do this once per copy of the bundle.


================================================================
  Mac — every time you want to launch
================================================================

1. Double-click "run.command" inside the "mac" folder.
2. A terminal window opens. Wait ~10 seconds.
3. Your default browser will open at http://localhost:8501 with the GUI.

To stop the app: close the terminal window.


================================================================
  Windows — every time you want to launch
================================================================

1. Open the "windows" folder.
2. Double-click "run.bat".
3. A command-prompt window opens. Wait ~10 seconds.
4. Your default browser will open at http://localhost:8501 with the GUI.

To stop the app: close the command-prompt window.


================================================================
  Using the GUI
================================================================

Once the GUI is open in your browser:

  1. In the sidebar, either click "Load PJME sample" for an instant demo,
     or upload your own CSV (must have a datetime column and a numeric
     value column).
  2. Pick the datetime and value columns from the dropdowns.
  3. Choose a forecast horizon (default: 720 hours = 30 days).
  4. Pick a model:
       - SARIMA   (~1 minute, classical statistical baseline)
       - LSTM     (~3-5 minutes, deep-learning model, more accurate)
  5. Click "Run forecast". Watch the live status box.
  6. The forecast chart appears in the main panel.
  7. Click "Download forecast CSV" to export the predictions.


================================================================
  Troubleshooting
================================================================

Mac: "run.command can't be opened because it is from an unidentified developer"
  → You skipped first_time_setup.command. Run that first.

Mac: terminal opens but browser does not
  → Manually open http://localhost:8501 in any browser.

Windows: SmartScreen warns about "unrecognized app"
  → Click "More info" → "Run anyway".

Forecast takes longer than expected
  → SARIMA is fastest. LSTM trains a model on your data, which takes
     several minutes. Be patient on the first click — subsequent clicks
     are also fresh trainings (no caching between runs).

================================================================
README

ok "README.txt written"

# ── step 7: clean up __pycache__ ──────────────────────────────────────────────
log "Removing __pycache__ and .pyc files"
find "$PORTABLE" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PORTABLE" -name "*.pyc" -delete 2>/dev/null || true
ok "Cache files removed"

# ── step 8: summary ───────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Build complete"
echo "================================================================"
echo "  mac/      $(size "$MAC_OUT")"
echo "  windows/  $(size "$WIN_OUT")"
echo "  Total     $(size "$PORTABLE")"
echo ""
echo "  To test the Mac bundle:"
echo "    bash '$MAC_OUT/first_time_setup.command' 2>/dev/null || true"
echo "    bash '$MAC_OUT/run.command'"
echo ""
echo "  Copy the entire dist/portable/ folder to a USB stick to distribute."
echo "================================================================"

#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
PYINSTALLER_BIN="${PYINSTALLER_BIN:-$VENV_PATH/bin/pyinstaller}"

if [[ ! -x "$PYINSTALLER_BIN" ]]; then
  echo "PyInstaller not found at $PYINSTALLER_BIN" >&2
  echo "Create or activate a Linux virtual environment before running this script." >&2
  exit 1
fi

pushd "$PROJECT_ROOT" >/dev/null
"$PYINSTALLER_BIN" --noconfirm GradientTemperature.spec
popd >/dev/null

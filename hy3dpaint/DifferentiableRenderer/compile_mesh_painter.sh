#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CXX_BIN="${CXX:-c++}"

# Use the active python environment to avoid ABI mismatch.
if command -v "${PYTHON_BIN}-config" >/dev/null 2>&1; then
  EXT_SUFFIX="$("${PYTHON_BIN}-config" --extension-suffix)"
else
  EXT_SUFFIX="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
fi

"${CXX_BIN}" -O3 -Wall -shared -std=c++11 -fPIC \
  $("${PYTHON_BIN}" -m pybind11 --includes) \
  mesh_inpaint_processor.cpp \
  -o "mesh_inpaint_processor${EXT_SUFFIX}"
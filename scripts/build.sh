#!/bin/bash
set -e
cd "$(dirname "$0")/.."

CORE="src/core/envelope.cpp src/core/calibration.cpp src/core/compiler.cpp src/core/scheduler.cpp src/core/waveform.cpp"
SIM="src/backends/simulator/statevector.cpp src/backends/mock/mock_awg.cpp"

# CLI
g++ -std=c++20 -I include -I include/qhal \
    src/cli/qhalc.cpp $CORE $SIM \
    -o /tmp/qhalc

# Shared C library
g++ -std=c++20 -fPIC -shared -I include -I include/qhal \
    src/bindings/c_abi.cpp $CORE $SIM \
    -o /tmp/libqhal.so

# Python extension
PYBIND_INC=$(python -c "import pybind11; print(pybind11.get_include())")
PY_INC=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PY_EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

g++ -std=c++20 -O2 -fPIC -shared \
    -I include -I include/qhal \
    -I "$PYBIND_INC" -I "$PY_INC" \
    src/bindings/pybind.cpp $CORE $SIM \
    -o python/qhal/qhal_cpp$PY_EXT

echo "built: /tmp/qhalc, /tmp/libqhal.so, python/qhal/qhal_cpp$PY_EXT"

#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/lib/ccache:$PATH
export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
export MAKEFLAGS=${MAKEFLAGS:--j8}

# Cache cleanup (optional)
mkdir -p /root/.cache/fenics /root/.cache/pip
chown root:root /root/.cache/pip || true
rm -f /root/.cache/fenics/* /tmp/call_basix* /work/*_petsc_* 2>/dev/null || true
rm -rf /work/__pycache__ 2>/dev/null || true

# Mark git dirs safe (only if you need git operations from inside container)
git config --global --add safe.directory /work/ufl-custom 2>/dev/null || true
git config --global --add safe.directory /work/ffcx-custom 2>/dev/null || true
git config --global --add safe.directory /work 2>/dev/null || true

# Editable installs are best for dev work
pushd /work/ufl-custom >/dev/null
python3 -m pip install -v -e . --no-deps
popd >/dev/null

pushd /work/ffcx-custom >/dev/null
# if you really need this checkout every time:
git checkout august/customquad
python3 -m pip install -v -e . --no-deps
popd >/dev/null

pushd /work >/dev/null
python3 -m pip install -v -e . -U
popd >/dev/null

echo "Installed ufl-custom, ffcx-custom, customquad (editable)."

#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/lib/ccache:$PATH
export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
export MAKEFLAGS=${MAKEFLAGS:--j8}

# Keep caches tidy (optional)
mkdir -p /root/.cache/fenics /root/.cache/pip
rm -f /root/.cache/fenics/* /tmp/call_basix* ./*_petsc_* 2>/dev/null || true
rm -rf ./__pycache__ 2>/dev/null || true

# Mark repo dirs safe (helps when mounted as root-owned)
git config --global --add safe.directory /work 2>/dev/null || true
git config --global --add safe.directory /work/ufl-custom 2>/dev/null || true
git config --global --add safe.directory /work/ffcx-custom 2>/dev/null || true

# Editable installs for dev speed
pushd /work/ufl-custom >/dev/null
python3 -m pip install -v -e . --no-deps
popd >/dev/null

pushd /work/ffcx-custom >/dev/null
# If you always want this branch, either keep this checkout,
# or set the submodule to track it and remove this line.
git checkout august/customquad
python3 -m pip install -v -e . --no-deps
popd >/dev/null

pushd /work >/dev/null
python3 -m pip install -v -e . -U
popd >/dev/null

echo "install-all complete."

#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/lib/ccache:$PATH
export CC=${CC:-/usr/lib/ccache/gcc}
export CXX=${CXX:-/usr/lib/ccache/g++}
export MAKEFLAGS=${MAKEFLAGS:--j8}
export PYTHONNOUSERSITE=1

# Sanity checks
if [ ! -d /work/ufl-custom/ufl ]; then
  echo "ERROR: /work/ufl-custom/ufl not found"
  exit 1
fi
if [ ! -d /work/ffcx-custom/ffcx ]; then
  echo "ERROR: /work/ffcx-custom/ffcx not found"
  exit 1
fi

# Mark git dirs safe
git config --global --add safe.directory /work 2>/dev/null || true
git config --global --add safe.directory /work/ufl-custom 2>/dev/null || true
git config --global --add safe.directory /work/ffcx-custom 2>/dev/null || true

# Optional cache cleanup
mkdir -p /root/.cache/fenics /root/.cache/pip
rm -f /root/.cache/fenics/* 2>/dev/null || true
rm -f /tmp/call_basix* 2>/dev/null || true
rm -f /work/*_petsc_* 2>/dev/null || true
find /work -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# SITEPKG=$(python3 -c "import site; print(site.getsitepackages()[0])")
# PTH_FILE="${SITEPKG}/00-customquad-dev.pth"
# printf '%s\n' 'import sys; sys.path[:0]=["/work/ffcx-custom","/work/ufl-custom","/work"]' > "${PTH_FILE}"
# echo "Wrote ${PTH_FILE}"

# Remove installed copies in the primary site-packages to avoid ambiguity.
# Your dev versions will come from /work via sys.path, and/or from editable installs below.
python3 -m pip uninstall -y ufl ffcx 2>/dev/null || true

# Install editable (useful for metadata/entry points; precedence is already handled by .pth)
echo "Installing ufl-custom (editable include -e)"
cd /work/ufl-custom
python3 -m pip install -v . --no-deps
cd /work

echo "Installing ffcx-custom (editable include -e)"
cd /work/ffcx-custom
git checkout august/customquad
python3 -m pip install -v . --no-deps
cd /work

echo "Installing customquad (editable)"
python3 -m pip install -v -e . -U

# echo
# echo "sys.path (first 15):"
# python3 -c "import sys; print('\n'.join(sys.path[:15]))"

# echo
# echo "Resolved import locations:"
# python3 -c "import dolfinx, ufl, ffcx; print('dolfinx:', dolfinx.__file__); print('ufl:', ufl.__file__); print('ffcx:', ffcx.__file__)"

echo
echo "install-all complete"

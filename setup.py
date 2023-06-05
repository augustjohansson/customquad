from setuptools import setup

VERSION = "0.6.0"
REQUIREMENTS = []

setup(name='customquad',
      version=VERSION,
      author='August Johansson',
      description='Custom quadrature in FEniCSx',
      packages=['customquad'],
      install_requires=REQUIREMENTS,
      zip_safe=False)

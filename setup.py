from setuptools import setup
from setuptools.command.install import install
import os
import shutil

VERSION = "0.6.0"
REQUIREMENTS = []


class PostInstall(install):
    def run(self):
        print("Starting PostInstall")
        install.run(self)
        print("Copy file")
        hfile = "./customquad/call_basix.h"
        target = "/usr/include"
        shutil.copy(hfile, target)


setup(
    name="customquad",
    version=VERSION,
    author="August Johansson",
    description="Custom quadrature in FEniCSx",
    packages=["customquad"],
    install_requires=REQUIREMENTS,
    cmdclass={"install": PostInstall},
)

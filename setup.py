import subprocess
import shutil
from setuptools import setup
from setuptools.command.install import install


VERSION = "0.6.0"
REQUIREMENTS = []


class PostInstall(install):
    def run(self):
        print("Starting PostInstall")
        install.run(self)

        print("Copy file")
        hfile = "./customquad/call_basix.h"
        target = "/usr/local/include"
        shutil.copy(hfile, target)

        print("Compile")
        compiler = "$CXX"
        flags = "-std=c++20 -fPIC -shared"
        cppfile = "./customquad/call_basix.cpp"
        sofile = "/usr/local/lib/libcall_basix.so"
        libs = "-lbasix"
        cmd = f"{compiler} {flags} {cppfile} -o {sofile} {libs}"
        print(cmd)
        subprocess.check_call(cmd, shell=True)
        print("Finished PostInstall")


setup(
    name="customquad",
    version=VERSION,
    author="August Johansson",
    description="Custom quadrature in FEniCSx",
    packages=["customquad"],
    install_requires=REQUIREMENTS,
    cmdclass={"install": PostInstall},
)

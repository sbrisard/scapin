import os.path

import pybind11
import setuptools

SCAPIN_INCLUDE_DIR = os.path.join("..", "include")


def get_metadata(name):
    with open(os.path.join("..", "metadata", name + ".txt")) as f:
        return f.read().strip()


def pybind11_extension(package_name, module_name):
    return setuptools.Extension(
        ".".join([package_name, module_name]),
        include_dirs=[pybind11.get_include(), SCAPIN_INCLUDE_DIR],
        sources=[os.path.join(package_name, module_name+".cpp")],
    )


if __name__ == "__main__":
    project_name = "pyscapin"

    with open(os.path.join("..", "README.md"), "r") as f:
        long_description = f.read()

    hooke = pybind11_extension(project_name, "hooke")

    setuptools.setup(
        name=project_name,
        version=get_metadata("version"),
        author=get_metadata("author"),
        author_email=get_metadata("author_email"),
        description="A framework for FFT-based, full-field numerical simulation of heterogeneous materials",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url=get_metadata("url"),
        packages=setuptools.find_packages(),
        ext_modules=[hooke],
    )

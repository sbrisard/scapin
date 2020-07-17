import os.path

import pybind11
import setuptools

SCAPIN_INCLUDE_DIR = os.path.join("..", "include")


def get_metadata(name):
    with open(os.path.join("..", "metadata", name + ".txt")) as f:
        return f.read().strip()


def pybind11_extension(module_name, metadata):
    return setuptools.Extension(
        ".".join([metadata["name"], module_name]),
        include_dirs=[pybind11.get_include(), SCAPIN_INCLUDE_DIR],
        sources=[os.path.join(metadata["name"], module_name + ".cpp")],
        define_macros=[
            ("__SCAPIN_VERSION__", r'\"' + metadata["version"] + r'\"'),
            ("__SCAPIN_AUTHOR__", r'\"' + metadata["author"] + r'\"'),
        ],
    )


if __name__ == "__main__":
    metadata = {
        "name": "pyscapin",
        "version": get_metadata("version"),
        "author": get_metadata("author"),
        "author_email": get_metadata("author_email"),
        "description": "A framework for FFT-based, full-field numerical simulation of heterogeneous materials",
        "url": get_metadata("url"),
    }

    with open(os.path.join("..", "README.md"), "r") as f:
        metadata["long_description"] = f.read()

    hooke = pybind11_extension("hooke", metadata)

    setuptools.setup(
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        ext_modules=[hooke],
        **metadata
    )

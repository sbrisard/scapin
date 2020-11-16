import configparser
import os.path

import pybind11
import setuptools

SCAPIN_INCLUDE_DIR = os.path.join("..", "include")


def get_metadata(key):
    with open(os.path.join("..", "metadata", key + ".txt"), "r") as f:
        return f.read().strip()


def pybind11_extension(module_name, metadata):
    return setuptools.Extension(
        ".".join([metadata["name"], module_name]),
        include_dirs=[pybind11.get_include(), metadata["scapin_include_dir"]],
        sources=[os.path.join(metadata["name"], module_name + ".cpp")],
        # TODO: Check that this is necessary
        define_macros=[
            ("__SCAPIN_VERSION__", r"\"" + metadata["version"] + r"\""),
            ("__SCAPIN_AUTHOR__", r"\"" + metadata["author"] + r"\""),
            ("_USE_MATH_DEFINES", ""),
        ],
    )


if __name__ == "__main__":
    metadata = {
        "name": "scapin",
        "version": get_metadata("version"),
        "author": get_metadata("author"),
        "author_email": get_metadata("email"),
        "description": get_metadata("description"),
        "url": get_metadata("repository"),
    }

    with open(os.path.join("..", "README.md"), "r") as f:
        metadata["long_description"] = f.read()

    config = configparser.ConfigParser()
    config.read("setup.cfg")
    metadata["scapin_include_dir"] = config["scapin"].get("include_dir", "")

    hooke = pybind11_extension("hooke", metadata)
    ms94 = pybind11_extension("ms94", metadata)

    setuptools.setup(
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        ext_modules=[hooke, ms94],
        **metadata
    )

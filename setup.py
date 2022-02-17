import os

from setuptools import find_packages, setup
from hookandlook import __version__

# Package meta-data.
NAME = "hookandlook"
DESCRIPTION = "A tool helping to gather stats and run checks during training deep learning models " \
              "with Pytorch using hooks."
URL = "https://github.com/arsenyinfo/hookandlook"
REQUIRES_PYTHON = ">=3.7.0"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        lineiter = f.read().splitlines()
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name=NAME,
    version=__version__,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=[
        "Machine Learning",
        "Deep Learning",
        "Computer Vision",
        "PyTorch",
    ],
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=load_requirements("requirements.txt"),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

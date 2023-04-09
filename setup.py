from setuptools import setup
from setuptools import find_packages


def load(path):
    return open(path, 'r').read()

dataset_helper_version = '1.4.0'

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"
]


if __name__ == "__main__":
    setup(
        name="OCTraN Dataset Helper",
        version=dataset_helper_version,
        description="OCTraN Dataset Helper",
        long_description=load('README.md'),
        long_description_content_type='text/markdown',
        platforms="OS Independent",
        package_data={'dataset_helper': ['README.md']},
        packages=find_packages(exclude=['tests']),
        install_requires=["pandas"],
    )
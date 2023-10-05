from setuptools import setup, find_packages

name = "promptbook"
version = "0.1"
description = "A Python library for in-context learning"
author = "@me0em"
author_email = "to.asmarkov@gmail.com"
url = "https://github.com/me0em/promptbook"
license = "WTFPL"

# Define the list of packages to include in the distribution
packages = find_packages()

# External dependencies
install_requires = [
    "numpy",
    "pandas",
    "torch",
    "einops",
    "tqdm",
    "annoy"
]

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

# Create the setup configuration
setup(
    python_requires='>3.10.0',
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=author_email,
    url=url,
    license=license,
    packages=packages,
    install_requires=install_requires,
    # Add any additional metadata as needed, such as classifiers, entry_points, etc.
)

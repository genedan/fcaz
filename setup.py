import setuptools

from fcaz.constants import BUILD_VERSION

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fcaz",
    version=BUILD_VERSION,
    author="Gene Dan",
    author_email="genedan@gmail.com",
    description="Property and Casualty Actuarial Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/genedan/fcaz",
    project_urls={
        "Documentation": "https://genedan.com/fcaz/docs"
    },
    install_requires=['scipy'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.10.0',
)
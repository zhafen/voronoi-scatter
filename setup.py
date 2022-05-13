import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="voronoi-scatter",
    version="1.0",
    author="Zach Hafen",
    author_email="zachary.h.hafen@gmail.com",
    description="Scatterplot with voronoi label placement.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhafen/augment",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'descartes>=1.1.0',
        'matplotlib>=3.5.1',
        'numpy>=1.21.5',
        'scipy>=1.7.3',
        'setuptools>=61.2.0',
        'Shapely>=1.7.1',
        'tqdm>=4.64.0',
    ],
    py_modules=[ 'voronoi_scatter', ],
)

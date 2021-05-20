from setuptools import setup

setup(
    name="npanalyst",
    version="1.0.0",
    packages=["npanalyst"],
    entry_points={"console_scripts": ["npanalyst = npanalyst.cli:main",]},
    install_requires=[
        "pandas",
        "rtree",
        "numpy",
        "scipy",
        "pymzml==2.4.5",  # Latest version which works 2.4.6 is broken
        "networkx",
        "joblib",
	"matplotlib",
    ],
    python_requires="~=3.6",  # fstrings all over the place...
)

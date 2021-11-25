from setuptools import setup

import npanalyst

setup(
    name="npanalyst",
    version=npanalyst.__version__,
    packages=["npanalyst"],
    entry_points={
        "console_scripts": [
            "npanalyst = npanalyst.cli:cli",
        ]
    },
    install_requires=[
        "click",
        "joblib",
        # "matplotlib-base",
        "networkx",
        "numpy",
        "pandas",
        "pygraphviz",
        "pymzml==2.4.5",
        "python-louvain",
        "rtree",
        "scikit-learn",
        "scipy",
    ],
    python_requires=">=3.8",
)

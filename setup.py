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
    ],
    python_requires=">=3.7"
)

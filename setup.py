from setuptools import setup

setup(
    name="npanalyst",
    version="0.0.2",
    packages=["npanalyst"],
        entry_points={
        'console_scripts': [
            'npanalyst = npanalyst.cli:main',
        ]
        },
    install_requires=[
        'pandas',
        'rtree',
        'tqdm',
        'numpy',
        'scipy',
        'pymzml',
        'networkx',
    ],
    python_requires='~=3.6' #fstrings all over the place...
)

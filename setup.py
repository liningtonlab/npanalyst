from setuptools import setup

setup(
    name="metabolate",
    version="0.0.1",
    packages=["metabolate"],
        entry_points={
        'console_scripts': [
            'metabolate = metabolate.cli:main',
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

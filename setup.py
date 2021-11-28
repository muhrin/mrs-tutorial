# -*- coding: utf-8 -*-
from setuptools import setup

__author__ = 'Martin Uhrin, Thomas J. Hardin, Tess E. Smidt'
__license__ = 'GPLv3'

about = {}
with open('e3nn_invstutorial/version.py') as f:
    exec(f.read(), about)  # pylint: disable=exec-used

setup(
    name='e3nn-invstutorial',
    version=about['__version__'],
    description='Tutorial for demonstrating use of rotation invariants in e3nn',
    long_description=open('README.md').read(),
    url='https://github.com/muhrin/e3-invs-tutorial.git',
    author=__author__,
    author_email='martin.uhrin.10@ucl.ac.uk',
    license=__license__,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='machine learning, atomic descriptor, room-temperature ionic liquid',
    python_requires='>=3.6',
    install_requires=[
        'ase',
        'e3nn',
        'ipywidgets',
        'matplotlib',
        'numpy',
        'notebook',
        'plotly',
        'RISE',
        'scipy',
        'torch',
        'tqdm',
    ],
    packages=['e3nn_invstutorial'],
    include_package_data=True,
)

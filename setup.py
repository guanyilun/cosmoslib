#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages
# for building fortran packages
from numpy.distutils.core import setup, Extension
import numpy as np
# for building cython packages
from Cython.Build import cythonize


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy', 'matplotlib',
                'cython', 'cytoolz', 'healpy', 'configparser']

cython_modules = cythonize(Extension(
    name="cosmoslib.like._likelihood",
    sources=["cosmoslib/like/_likelihood.pyx"],
    include_dirs=[np.get_include()],
    # extra_compile_args=['-fopenmp'],
    # extra_link_args=['-fopenmp'],
))

f2py_modules = [
    Extension(
        name='cosmoslib.utils.cwignerd',
        sources=['deps/src/wignerd/wignerd.pyf','deps/src/wignerd/wignerd.c']
    )
]

ext_modules = cython_modules + f2py_modules

setup(
    author="Yilun Guan",
    author_email='zoom.aaron@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Some reusable codes in my cosmology research",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cosmoslib',
    name='cosmoslib',
    packages=find_packages(include=['cosmoslib']),
    url='https://github.com/guanyilun/cosmoslib',
    version='0.1.0',
    zip_safe=False,
    ext_modules=ext_modules
)

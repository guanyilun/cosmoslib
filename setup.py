#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
# from distutils.extension import Extension
# from Cython.Build import cythonize
# import numpy as np

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy', 'matplotlib', 'cython']

setup(
    author="Yilun Guan",
    author_email='zoom.aaron@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
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
    # not used build by default
    # ext_modules=cythonize([
    #     Extension(
    #         name="cosmoslib.ps._ps",
    #         sources=["cosmoslib/ps/ps.pyx"],
    #         include_dirs=[np.get_include()],
    #         extra_compile_args=['-fopenmp'],
    #         extra_link_args=['-fopenmp'],
    #     )
    # ])
)

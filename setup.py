#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas', 'matplotlib', 'numpy', 'seaborn']

setup_requirements = ['pandas', 'matplotlib', 'numpy', 'seaborn']

test_requirements = ['pandas', 'matplotlib', 'numpy', 'seaborn']

setup(
    author="Jeff Herman",
    author_email='jherman1199@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python library to make interpretting machine learning models easier :) ",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ezinterpret',
    name='ezinterpret',
    packages=find_packages(include=['ezinterpret', 'ezinterpret.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/sik-flow/ezinterpret',
    version='0.1.9',
    zip_safe=False,
)

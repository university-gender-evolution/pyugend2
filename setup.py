#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="krishnab bhogaonker",
    author_email='cyclotomiq@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Gender balance in university project. Mark 2.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    package_dir={'pyugend2':
                 'pyugend2'},
    include_package_data=True,
    keywords='pyugend2',
    name='pyugend2',
    packages=find_packages(include=['pyugend2']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/00krishna/pyugend2',
    version='0.1.0',
    zip_safe=False,
)

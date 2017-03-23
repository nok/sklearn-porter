# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages


path = os.path.abspath(os.path.dirname(__file__))

# Version:
version_path = str(os.path.join(path, 'sklearn_porter', '__version__.txt'))
version = open(version_path).readlines().pop()
if isinstance(version, bytes):
    version = version.decode('utf-8')
version = str(version).strip()

# Requirements:
requirements_path = os.path.join(path, 'requirements.txt')
with open(requirements_path) as f:
    all_reqs = f.read().split('\n')
requirements = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name='sklearn-porter',
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    version=version,
    description='Transpile trained scikit-learn models to C, Java, JavaScript and others.',
    author='Darius Morawiec',
    author_email='ping@nok.onl',
    url='https://github.com/nok/sklearn-porter/tree/stable',
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=requirements,
    keywords=['sklearn', 'scikit-learn'],
    license='MIT',
)

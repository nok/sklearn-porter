# -*- coding: utf-8 -*-

import os
from setuptools import setup
from setuptools import find_packages


def read_version():
    src_dir = os.path.abspath(os.path.dirname(__file__))
    ver_file = os.path.join(src_dir, 'sklearn_porter', '__version__.txt')
    version = open(ver_file, 'r').readlines().pop()
    if isinstance(version, bytes):
        version = version.decode('utf-8')
    version = str(version).strip()
    return version


def parse_requirements():
    src_dir = os.path.abspath(os.path.dirname(__file__))
    req_file = os.path.join(src_dir, 'requirements.txt')
    reqs = open(req_file, 'r').read().strip().split('\n')
    reqs = [req.strip() for req in reqs if 'git+' not in req]
    return reqs


setup(
    name='sklearn-porter',
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    version=read_version(),
    description='Transpile trained scikit-learn models to C, Java, JavaScript and others.',
    author='Darius Morawiec',
    author_email='ping@nok.onl',
    url='https://github.com/nok/sklearn-porter/tree/stable',
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=parse_requirements(),
    keywords=['sklearn', 'scikit-learn'],
    license='MIT',
)

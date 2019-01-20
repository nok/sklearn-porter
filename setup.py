# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

from sklearn_porter import meta  # see sklearn_porter/package.json


setup(
    name=meta.get('name'),
    description=meta.get('description'),
    long_description=meta.get('long_description'),
    long_description_content_type='text/markdown',
    keywords=meta.get('keywords'),
    url=meta.get('url'),
    author=meta.get('author'),
    author_email=meta.get('author_email'),
    install_requires=meta.get('requirements'),
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'porter = sklearn_porter.cli.__main__:main'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    version=meta.get('version'),
    license=meta.get('license'),
)

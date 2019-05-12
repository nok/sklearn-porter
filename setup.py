# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import join
from json import load


def load_meta(path):
    """
    Load meta information from file setup.json.
    :param path: The path to setup.json
    :return: Dictionary of meta information.
    """
    with open(path, mode='r', encoding='utf-8') as f:
        meta = load(f, encoding='utf-8')
        meta = {k: v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in meta.items()}

        src_dir = abspath(dirname(path))

        if 'requirements' in meta and \
                str(meta['requirements']).startswith('file://'):
            reqs_path = str(meta['requirements'])[7:]
            reqs_path = join(src_dir, reqs_path)
            if exists(reqs_path):
                reqs = open(reqs_path, 'r', encoding='utf-8').read()
                reqs = reqs.strip().split('\n')
                reqs = [req.strip() for req in reqs if 'git+' not in req]
                meta['requirements'] = reqs

        if 'long_description' in meta and \
                str(meta['long_description']).startswith('file://'):
            readme_path = str(meta['long_description'])[7:]
            readme_path = join(src_dir, readme_path)
            if exists(readme_path):
                readme = open(readme_path, 'r', encoding='utf-8').read()
                readme = readme.strip()
                meta['long_description'] = readme

    return meta


def main():
    path = join(abspath(dirname(__file__)), 'sklearn_porter', 'setup.json')
    meta = load_meta(path)
    setup(
        name=meta.get('name'),
        description=meta.get('description'),
        long_description=meta.get('long_description'),
        long_description_content_type='text/markdown',
        keywords=meta.get('keywords'),
        url=meta.get('url'),
        author=meta.get('author'),
        author_email=meta.get('author_email'),
        install_requires=[
            'scikit-learn>=0.14.1'
        ],
        extras_require={
            'examples': [
                'jupyterlab>=0.33.12'
            ],
            'development': [
                'twine>=1.12.1',
                'pylint>=1.9.3',
                'pytest>=3.9.2',
                'jupytext>=0.8.3',
            ],
        },
        packages=find_packages(exclude=[
            'tests.*',
            'tests'
        ]),
        test_suite='pytest',
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'porter = sklearn_porter.cli.__main__:main'
            ],
        },
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
        ],
        version=meta.get('version'),
        license=meta.get('license'),
    )


if __name__ == '__main__':
    main()

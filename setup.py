# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import join
from json import load


def load_package_data(path):
    """Load meta data about this package from `package.json`.

    Parameters
    ----------
    path : str
        The path to file `package.json`.

    Returns
    -------
    meta : dict
        Dictionary of key value pairs.
    """
    with open(path) as f:
        meta = load(f, encoding='utf-8')
        meta = {k: v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in meta.items()}

        # Root of the source code `/sklearn_porter`:
        src_dir = abspath(dirname(path))

        # Load and parse requirements file:
        if 'requirements' in meta and \
                str(meta['requirements']).startswith('file://'):
            req_path = str(meta['requirements'])[7:]
            req_path = join(src_dir, req_path)
            if exists(req_path):
                reqs = open(req_path, 'r').read().strip().split('\n')
                reqs = [req.strip() for req in reqs if 'git+' not in req]
                meta['requirements'] = reqs

        # Load readme file:
        if 'long_description' in meta and \
                str(meta['long_description']).startswith('file://'):
            readme_path = str(meta['long_description'])[7:]
            readme_path = join(src_dir, readme_path)
            if exists(readme_path):
                readme = open(readme_path, 'r').read().strip()
                meta['long_description'] = readme

    return meta


def main():
    file_path = abspath(dirname(__file__))
    package_path = join(file_path, 'sklearn_porter', 'package.json')
    package = load_package_data(package_path)

    setup(
        name=package.get('name'),
        description=package.get('description'),
        long_description=package.get('long_description', ''),
        long_description_content_type='text/markdown',
        keywords=package.get('keywords'),
        url=package.get('url'),
        author=package.get('author'),
        author_email=package.get('author_email'),
        install_requires=package.get('requirements', []),
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
        version=package.get('version'),
        license=package.get('license'),
    )


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import join
from json import load


def _load_meta(path):
    """
    Load meta data about this package from file pypi.json.
    :param path: The path to pypi.json
    :return: Dictionary of key value pairs.
    """
    with open(path, 'r', encoding='utf-8') as f:
        meta = load(f)
        meta = {k: v.decode('utf-8') if isinstance(v, bytes) else str(v)
                for k, v in meta.items()}

        src_dir = abspath(dirname(path))

        if 'requirements' in meta and str(meta['requirements']).startswith('file://'):
            req_path = join(src_dir, meta['requirements'][7:])
            if exists(req_path):
                with open(req_path, 'r', encoding='utf-8') as req_file:
                    reqs = req_file.read().strip().split('\n')
                    reqs = [req.strip() for req in reqs if 'git+' not in req]
                    meta['requirements'] = reqs
            else:
                meta['requirements'] = ''

        if 'long_description' in meta and str(meta['long_description']).startswith('file://'):
            readme_path = join(src_dir, meta['long_description'][7:])
            if exists(readme_path):
                with open(readme_path, 'r', encoding='utf-8') as readme_file:
                    meta['long_description'] = readme_file.read().strip()
            else:
                meta['long_description'] = ''

    return meta


package = join(abspath(dirname(__file__)), 'sklearn_porter', 'pypi.json')
meta = _load_meta(package)


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

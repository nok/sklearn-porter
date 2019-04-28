# -*- coding: utf-8 -*-

from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import join
from json import load

from sklearn_porter.Porter import Porter


def load_meta(path):
    """
    Load meta information from file pypi.json.
    :param path: The path to pypi.json
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
                reqs = [req.strip() for req in reqs]
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


meta = load_meta(join(abspath(dirname(__file__)), 'pypi.json'))

__author__ = meta.get('author')
__email__ = meta.get('author_email')
__license__ = meta.get('license')
__version__ = meta.get('version')

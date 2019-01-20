# -*- coding: utf-8 -*-

from os.path import abspath
from os.path import dirname
from os.path import join
from json import load


def _load_meta(path):
    """
    Load meta data about this package from file package.json.
    :param path: The path to package.json
    :return: Dictionary of key value pairs.
    """
    with open(path) as f:
        meta = load(f, encoding='utf-8')
        meta = {k: v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in meta.items()}

        src_dir = abspath(dirname(path))

        if 'requirements' in meta and \
                str(meta['requirements']).startswith('file://'):
            req_path = str(meta['requirements'])[7:]
            req_path = join(src_dir, req_path)
            reqs = open(req_path, 'r').read().strip().split('\n')
            reqs = [req.strip() for req in reqs if 'git+' not in req]
            meta['requirements'] = reqs

        if 'long_description' in meta and \
                str(meta['long_description']).startswith('file://'):
            readme_path = str(meta['long_description'])[7:]
            readme_path = join(src_dir, readme_path)
            readme = open(readme_path, 'r').read().strip()
            meta['long_description'] = readme

    return meta


package = join(abspath(dirname(__file__)), 'package.json')
meta = _load_meta(package)

__author__ = meta.get('author')
__email__ = meta.get('author_email')
__license__ = meta.get('license')
__version__ = meta.get('version')

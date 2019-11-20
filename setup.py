# -*- coding: utf-8 -*-

from os.path import abspath, dirname, exists, join
from setuptools import find_packages, setup
from sys import version_info

from sklearn_porter import __author__, __email__, __license__, __version__


def _check_python_version():
    """Check the used Python version."""
    if version_info[:2] < (3, 5):
        msg = 'The used Python version is not ' \
              'supported, please use Python >= 3.5'
        raise RuntimeError(msg)


def _read_file(path):
    """
    Read the content from a text file.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    Return the content as string.
    """
    content = ''
    if exists(path):
        content = open(path, 'r', encoding='utf-8').read().strip()
    return content


def main():
    _check_python_version()

    name = 'sklearn-porter'
    desc = 'Transpile trained scikit-learn models ' \
           'to C, Java, JavaScript and others.'

    readme_path = join(abspath(dirname(__file__)), 'readme.md')
    long_desc = _read_file(readme_path)

    setup(
        name=name,
        description=desc,
        long_description=long_desc,
        long_description_content_type='text/markdown',
        keywords=['sklearn', 'scikit-learn'],
        url='https://github.com/nok/sklearn-porter',
        install_requires=[
            'scikit-learn>=0.17',
            'Jinja2>=2.10.1',
            'loguru>=0.3.2',
        ],
        extras_require={
            'examples': ['notebook==5.*'],
            'development': [
                'twine>=1.12.1',
                'pylint>=1.9.3',
                'pytest>=3.9.2',
                'pytest-xdist>=1.29.0',
                'jupytext>=0.8.3',
            ],
        },
        packages=find_packages(exclude=['tests.*', 'tests']),
        test_suite='pytest',
        include_package_data=True,
        entry_points={
            'console_scripts': ['porter = sklearn_porter.cli.__main__:main'],
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
        author=__author__,
        author_email=__email__,
        version=__version__,
        license=__license__,
    )


if __name__ == '__main__':
    main()

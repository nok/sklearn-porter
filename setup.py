# -*- coding: utf-8 -*-

from os.path import abspath, dirname, exists, join
from setuptools import find_packages, setup
from sys import version_info


def _check_python_version():
    """Check the used Python version."""
    if version_info[:2] < (3, 5):
        msg = 'The used Python version is not ' \
              'supported, please use Python >= 3.5'
        raise RuntimeError(msg)


def _read_text(path):
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

    file_dir = abspath(dirname(__file__))

    # Read readme.md
    path_readme = join(file_dir, 'readme.md')
    long_desc = _read_text(path_readme)

    # Read __version__.txt
    path_version = join(file_dir, 'sklearn_porter', '__version__.txt')
    version = _read_text(path_version)

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
            'joblib',
        ],
        extras_require={
            'examples': ['notebook==5.*'],
            'development': [
                'codecov>=2.0.15',
                'twine>=1.12.1',
                'pylint>=1.9.3',
                'pytest>=3.9.2',
                'pytest-cov>=2.7.1',
                'pytest-sugar>=0.9.2',
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
        author='Darius Morawiec',
        author_email='nok@users.noreply.github.com',
        license='MIT',
        version=version,
    )


if __name__ == '__main__':
    main()

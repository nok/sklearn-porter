from argparse import SUPPRESS, ArgumentParser

from sklearn_porter import __version__ as porter_version


def arg_help(p: ArgumentParser):
    p.add_argument(
        '-h', '--help', action='help', help='Show this help message and exit.'
    )


def arg_version(p: ArgumentParser):
    p.add_argument(
        '-v',
        '--version',
        action='version',
        version=str(porter_version),
        help='Show the version number and exit.',
    )


def arg_skip_warnings(p: ArgumentParser):
    p.add_argument(
        '--skip-warnings',
        action='store_true',
        default=False,
        help='Ignore and skip raised warnings.',
    )


def arg_ignore_preprocessing(p: ArgumentParser):
    p.add_argument(
        '--ignore-preprocessing',
        action='store_true',
        default=False,
        help=(
            'Port only the final estimator of a pipeline\n'
            'and ignore its preprocessing steps.'
        ),
    )


def arg_class_name(p: ArgumentParser):
    p.add_argument(
        '--class-name',
        type=str,
        required=False,
        help=(
            'The class name of the ported estimator, which is\n'
            'also the name of the generated files.'
        ),
    )


def arg_debug(p: ArgumentParser):
    p.add_argument('--debug', action='store_true', help=SUPPRESS)

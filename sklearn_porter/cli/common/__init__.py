from argparse import SUPPRESS, ArgumentParser

from sklearn_porter import __version__ as porter_version


def arg_help(p: ArgumentParser):
    p.add_argument(
        '-h',
        '--help',
        help="Show this help message and exit."
    )


def arg_version(p: ArgumentParser):
    p.add_argument(
        '-v',
        '--version',
        action='version',
        version=str(porter_version),
        help='Show the version number and exit.'
    )


def arg_skip_warnings(p: ArgumentParser):
    p.add_argument(
        '--skip-warnings',
        action='store_true',
        default=False,
        help='Ignore and skip raised warnings.'
    )


def arg_debug(p: ArgumentParser):
    p.add_argument('--debug', action='store_true', help=SUPPRESS)


def arg_json(p: ArgumentParser):
    p.add_argument(
        '--json',
        required=False,
        default=False,
        action='store_true',
        help='Return result in JSON format.'
    )

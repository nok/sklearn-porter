from argparse import RawTextHelpFormatter, _SubParsersAction
from logging import DEBUG
from textwrap import dedent
from typing import Dict

# sklearn-porter
from sklearn_porter import options, show
from sklearn_porter.cli.common import arg_debug, arg_help
from sklearn_porter.language import LANGUAGES


def config(sub_parser: _SubParsersAction):
    header = 'The subcommand `show` lists all supported ' \
             'estimators and programming languages.'
    footer = dedent("""
        Examples:
          `porter show`
    """)

    parser = sub_parser.add_parser(
        'show',
        description=header,
        help='Show the supported estimators and programming languages.',
        epilog=footer,
        formatter_class=RawTextHelpFormatter,
        add_help=False,
    )
    for group in parser._action_groups:
        group.title = str(group.title).capitalize()
    parser.add_argument(
        '-l',
        '--language',
        type=str,
        required=False,
        choices=LANGUAGES.keys(),
        help='The name of the programming language.'
    )
    for fn in (arg_debug, arg_help):
        fn(parser)

    parser.set_defaults(func=main)


def main(args: Dict, silent: bool = False) -> str:
    if args.get('debug'):
        options['logging.level'] = DEBUG

    out = show(args.get('language'))

    if not silent:
        print(out)

    return out

# -*- coding: utf-8 -*-

import sys
from os import environ
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from sklearn_porter import __version__ as porter_version
from sklearn_porter.cli.command import port, show
from sklearn_porter.cli.common import arg_help, arg_version


def parse_args(args):
    header = 'sklearn-porter CLI v{}'.format(porter_version)
    footer = dedent(
        """
        Examples:
          `porter show`
          `porter port model.pkl --language js --template attached`
        
        Manuals:
          https://github.com/nok/sklearn-porter
          https://github.com/scikit-learn/scikit-learn
    """
    )
    parser = ArgumentParser(
        description=header,
        formatter_class=RawTextHelpFormatter,
        add_help=False,
        epilog=footer,
    )
    for group in parser._action_groups:
        group.title = str(group.title).capitalize()

    sp = parser.add_subparsers(
        metavar='command',
        dest='cmd',
    )
    sp.required = True

    show.config(sp)
    port.config(sp)

    arg_version(parser)
    arg_help(parser)

    if len(sys.argv) == 1 and 'SKLEARN_PORTER_PYTEST' not in environ:
        parser.print_help(sys.stdout)
        sys.exit(1)

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    if hasattr(args, 'func'):
        func = args.func
        delattr(args, 'func')
        func(vars(args))


if __name__ == "__main__":
    main()

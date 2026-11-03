import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from textwrap import dedent

from sklearn_porter import __version__ as porter_version
from sklearn_porter.cli.command import port, save, show
from sklearn_porter.cli.common import arg_help, arg_version


def parse_args(args):
    header = f'sklearn-porter CLI v{porter_version}'
    footer = dedent(
        """
        Examples:
          `porter show`
          `porter port model.joblib --language js --template attached`
          `porter save model.joblib --directory /tmp --skip-warnings`

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
    save.config(sp)

    arg_version(parser)
    arg_help(parser)

    if not args:
        parser.print_help(sys.stdout)
        sys.exit(1)

    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    if hasattr(args, 'func'):
        func = args.func
        delattr(args, 'func')
        try:
            func(vars(args))
        except Exception as e:
            # Show the traceback only with `--debug`:
            if getattr(args, 'debug', False):
                raise
            sys.exit(f'Error: {e}')


if __name__ == '__main__':
    main()

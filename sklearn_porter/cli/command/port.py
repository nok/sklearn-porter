from argparse import RawTextHelpFormatter
from logging import DEBUG
from pathlib import Path
from textwrap import dedent

# sklearn-porter
from sklearn_porter import Porter, options
from sklearn_porter.cli.common import (
    arg_debug,
    arg_help,
    arg_ignore_preprocessing,
    arg_skip_warnings,
)
from sklearn_porter.cli.utils import load_model
from sklearn_porter.language import LANGUAGES


def config(sub_parser):
    header = (
        'The subcommand `port` transpiles a trained '
        'estimator (or transformer) to a specific programming language.'
    )
    footer = dedent(
        """
        Examples:
          `porter port model.pkl --language js --template attached`
          `porter port model.pkl --skip-warnings`
          `porter port pipeline.pkl --ignore-preprocessing`
    """
    )

    parser = sub_parser.add_parser(
        'port',
        description=header,
        help='Port a trained estimator to a programming language.',
        epilog=footer,
        formatter_class=RawTextHelpFormatter,
        add_help=False,
    )
    for group in parser._action_groups:
        group.title = str(group.title).capitalize()

    parser.add_argument(
        'model', type=str, help='Path to an exported estimator.'
    )
    parser.add_argument(
        '-l',
        '--language',
        type=str,
        required=False,
        choices=LANGUAGES.keys(),
        help='The name of the programming language.',
    )
    parser.add_argument(
        '-t',
        '--template',
        type=str,
        required=False,
        choices=['attached', 'combined', 'exported'],
        help='The name of the template.',
    )

    for fn in (
        arg_ignore_preprocessing,
        arg_skip_warnings,
        arg_debug,
        arg_help,
    ):
        fn(parser)

    parser.set_defaults(func=main)


def main(args: dict, silent: bool = False):
    if args.get('debug'):
        options['logging.level'] = DEBUG

    path = Path(args['model'])
    mdl = load_model(path, args.get('skip_warnings', False))
    est = Porter(
        mdl, ignore_preprocessing=args.get('ignore_preprocessing', False)
    )

    if language := args.get('language'):
        est.language = language

    if template := args.get('template'):
        est.template = template

    out = est.port()
    if isinstance(out, list):  # the ported steps of a pipeline
        out = '\n\n'.join(
            o if isinstance(o, str) else '\n\n'.join(o) for o in out
        )

    if not silent:
        print(out)

    return out

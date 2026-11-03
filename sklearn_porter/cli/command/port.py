from argparse import RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

# sklearn-porter
from sklearn_porter import Porter
from sklearn_porter.cli.common import (
    arg_class_name,
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
          `porter port model.joblib --language js --template attached`
          `porter port model.joblib --skip-warnings`
          `porter port pipeline.joblib --ignore-preprocessing`
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
        arg_class_name,
        arg_ignore_preprocessing,
        arg_skip_warnings,
        arg_debug,
        arg_help,
    ):
        fn(parser)

    parser.set_defaults(func=main)


def main(args: dict, silent: bool = False):
    path = Path(args['model'])
    mdl = load_model(path, args.get('skip_warnings', False))
    est = Porter(
        mdl,
        class_name=args.get('class_name'),
        ignore_preprocessing=args.get('ignore_preprocessing', False),
    )

    if language := args.get('language'):
        est.language = language

    if template := args.get('template'):
        est.template = template

    # The standard output can only contain one file:
    if est.template == 'exported':
        msg = (
            'The template `exported` creates two files (the source code and '
            'the model data), save them with `porter save` instead.'
        )
        raise ValueError(msg)

    out = est.port()

    if not silent:
        print(out)

    return out

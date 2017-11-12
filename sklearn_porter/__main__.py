# -*- coding: utf-8 -*-

import os
import sys
import argparse

from . import Porter


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            'Transpile trained scikit-learn estimators '
            'to C, Java, JavaScript and others. '),
        epilog=(
            'More details on: '
            'https://github.com/nok/sklearn-porter'))
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    required.add_argument(
        '--input', '-i',
        required=True,
        help=(
            'Path to an exported estimator '
            'in pickle (.pkl) format.'))
    optional.add_argument(
        '--output', '-o',
        required=False,
        help=(
            'Path to the destination directory '
            'where the transpiled estimator will be '
            'stored.'))
    optional.add_argument(
        '--class_name',
        default='Brain',
        required=False,
        help='Define the class name in the final output.')
    optional.add_argument(
        '--method_name',
        default='predict',
        required=False,
        help='Define the method name in the final output.')
    optional.add_argument(
        '--pipe', '-p',
        required=False,
        default=False,
        action='store_true',
        help='Print the transpiled estimator to the console.')
    languages = {
        'c': 'C',
        'java': 'Java',
        'js': 'JavaScript',
        'go': 'Go',
        'php': 'PHP',
        'ruby': 'Ruby'
    }
    optional.add_argument(
        '--language', '-l',
        choices=languages.keys(),
        default='java',
        required=False,
        help=argparse.SUPPRESS)
    for key, lang in list(languages.items()):
        optional.add_argument(
            '--{}'.format(key),
            action='store_true',
            help='Set {} as the target programming language.'.format(lang))
    args = vars(parser.parse_args(args))
    return args


def main():
    args = parse_args(sys.argv[1:])

    # Check input data:
    input_path = str(args.get('input'))
    if not input_path.endswith('.pkl') or not os.path.isfile(input_path):
        error = 'No valid estimator in pickle format was found.'
        sys.exit('Error: {}'.format(error))

    # Load data:
    from sklearn.externals import joblib
    estimator = joblib.load(input_path)

    # Determine the target programming language:
    language = str(args.get('language'))  # with default language
    languages = ['c', 'java', 'js', 'go', 'php', 'ruby']
    for key in languages:
        if args.get(key):  # found explicit assignment
            language = key
            break

    # Port estimator:
    try:
        porter = Porter(estimator, language=language)
        class_name = str(args.get('class_name'))
        method_name = str(args.get('method_name'))
        output = porter.export(class_name=class_name,
                               method_name=method_name,
                               output=str(args.get('output')),
                               details=True)
    except Exception as e:
        sys.exit('Error: {}'.format(str(e)))
    else:
        # Print transpiled estimator to the console:
        if bool(args.get('pipe', False)):
            print(output.get('model'))
            sys.exit(0)

        # Define destination path:
        dest_dir = str(args.get('output'))
        filename = output.get('filename')
        if dest_dir != '' and os.path.isdir(dest_dir):
            dest_path = os.path.join(dest_dir, filename)
        else:
            dest_dir = input_path.split(os.sep)
            del dest_dir[-1]
            dest_dir += [filename]
            dest_path = os.sep.join(dest_dir)

        # Save transpiled estimator:
        with open(dest_path, 'w') as file_:
            file_.write(output.get('model'))


if __name__ == "__main__":
    main()

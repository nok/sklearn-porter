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
    parser.add_argument(
        '--input', '-i',
        required=True,
        help=(
            'Set the path of an exported model '
            'in pickle (.pkl) format.'))
    parser.add_argument(
        '--output', '-o',
        required=False,
        help=(
            'Set the destination directory, '
            'where the transpiled estimator will be '
            'stored.'))
    languages = {
        'c': 'C',
        'java': 'Java',
        'js': 'JavaScript',
        'go': 'Go',
        'php': 'PHP',
        'ruby': 'Ruby'
    }
    parser.add_argument(
        '--language', '-l',
        choices=languages.keys(),
        default='java',
        required=False,
        help=(
            'Set the target programming language '
            '({}).'.format(', '.join(['"{}"'.format(key)
                                      for key in languages.keys()]))))
    for key, lang in list(languages.items()):
        parser.add_argument(
            '--{}'.format(key),
            action='store_true',
            help='Set {} as the target programming language.'.format(lang))
    args = vars(parser.parse_args(args))
    return args


def main():
    args = parse_args(sys.argv[1:])

    input_path = str(args['input'])
    if input_path.endswith('.pkl') and os.path.isfile(input_path):

        # Load data:
        from sklearn.externals import joblib
        model = joblib.load(input_path)

        # Determine the target programming language:
        language = str(args['language'])  # with default language
        languages = ['c', 'java', 'js', 'go', 'php', 'ruby']
        for key in languages:
            if args.get(key):  # found ecplicit assignment
                language = key
                break

        # Port model:
        porter = Porter(model, language=language)
        details = porter.export(details=True)
        filename = details.get('filename')

        # Define destination path:
        dest_dir = str(args['output'])
        if dest_dir != '' and os.path.isdir(dest_dir):
            dest_dir = str(args['output'])
            dest_path = os.path.join(dest_dir, filename)
        else:
            dest_dir = input_path.split(os.sep)
            del dest_dir[-1]
            dest_dir += [filename]
            dest_path = os.sep.join(dest_dir)

        # Save transpiled model:
        with open(dest_path, 'w') as file_:
            file_.write(details.get('model'))
    else:
        raise ValueError('No valid model in pickle format was found.')

if __name__ == "__main__":
    main()

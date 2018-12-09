# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import os
import os.path
import argparse

from sklearn.externals import joblib

from sklearn_porter import __version__ as porter_version
from sklearn_porter.Porter import Porter
from sklearn_porter.language import *


header = '''
             #          
### ### ### ### ### ### 
# # # # #    #  ##  #   
### ### #    ## ### #   v{}
#

Transpile trained scikit-learn estimators
to C, Java, JavaScript and others.

Usage:
  porter --input INPUT [--output OUTPUT]
         [--class_name CLASS_NAME] [--method_name METHOD_NAME]
         [--export] [--checksum] [--data] [--pipe]
         [--c] [--java] [--js] [--go] [--php] [--ruby]
         [--help] [--version]'''.format(porter_version)


def parse_args(args):
    epilog = 'More details on https://github.com/nok/sklearn-porter'
    description = ''
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog, usage=header)

    # Remove the default arguments group:
    parser._action_groups.pop()

    # Required arguments:
    required = parser.add_argument_group('Required arguments')
    required.add_argument('--input', '-i',
                          required=True,
                          help=('Path to an exported estimator in pickle '
                                '(.pkl) format.'))

# Optional arguments:
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--output', '-o',
                          required=False,
                          help=('Path to the destination directory where the '
                                'transpiled estimator will be stored.'))
    optional.add_argument('--class_name',
                          default=None,
                          required=False,
                          help='Define the class name in the final output.')
    optional.add_argument('--method_name',
                          default='predict',
                          required=False,
                          help='Define the method name in the final output.')
    optional.add_argument('--export', '-e',
                          required=False,
                          default=False,
                          action='store_true',
                          help='Whether to export the model data or not.')
    optional.add_argument('--checksum', '-s',
                          required=False,
                          default=False,
                          action='store_true',
                          help='Whether to append the checksum to the '
                               'filename or not.')
    optional.add_argument('--data', '-d',
                          required=False,
                          default=False,
                          action='store_true',
                          help='Whether to export just the model data or all.')
    optional.add_argument('--pipe', '-p',
                          required=False,
                          default=False,
                          action='store_true',
                          help='Print the transpiled estimator to the console.')

    # Languages:
    langs = parser.add_argument_group('Programming languages')
    languages = {key: clazz.LABEL for key, clazz in list(LANGUAGES.items())}
    langs.add_argument('--language', '-l',
                       choices=languages.keys(),
                       default='java',
                       required=False,
                       help=argparse.SUPPRESS)
    for key, label in list(languages.items()):
        help = 'Set \'{}\' as the target programming language.'.format(label)
        langs.add_argument('--{}'.format(key), action='store_true', help=help)

    # Extra arguments:
    extras = parser.add_argument_group('Extra arguments')
    extras.add_argument('--version', '-v', action='version',
                        version='sklearn-porter v{}'.format(porter_version))

    # Show help by default:
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Return dictionary:
    args = vars(parser.parse_args(args))
    return args


def main():
    args = parse_args(sys.argv[1:])

    # Check input data:
    pkl_file_path = str(args.get('input'))
    if not os.path.isfile(pkl_file_path):
        exit_msg = 'No valid estimator in pickle ' \
                   'format was found at \'{}\'.'.format(pkl_file_path)
        sys.exit('Error: {}'.format(exit_msg))

    # Load data:
    estimator = joblib.load(pkl_file_path)

    # Determine the target programming language:
    language = str(args.get('language'))  # with default language
    languages = ['c', 'java', 'js', 'go', 'php', 'ruby']
    for key in languages:
        if args.get(key):  # found explicit assignment
            language = key
            break

    # Define destination path:
    dest_dir = str(args.get('output'))
    if dest_dir == '' or not os.path.isdir(dest_dir):
        dest_dir = pkl_file_path.split(os.sep)
        del dest_dir[-1]
        dest_dir = os.sep.join(dest_dir)

    # Port estimator:
    try:
        class_name = args.get('class_name')
        method_name = args.get('method_name')
        with_export = bool(args.get('export'))
        with_checksum = bool(args.get('checksum'))
        porter = Porter(estimator, language=language)
        output = porter.export(class_name=class_name, method_name=method_name,
                               export_dir=dest_dir, export_data=with_export,
                               export_append_checksum=with_checksum,
                               details=True)
    except Exception as exception:
        # Catch any exception and exit the process:
        sys.exit('Error: {}'.format(str(exception)))
    else:
        # Print transpiled estimator to the console:
        if bool(args.get('pipe', False)):
            print(output.get('estimator'))
            sys.exit(0)

        only_data = bool(args.get('data'))
        if not only_data:
            filename = output.get('filename')
            dest_path = dest_dir + os.sep + filename
            # Save transpiled estimator:
            with open(dest_path, 'w') as file_:
                file_.write(output.get('estimator'))


if __name__ == "__main__":
    main()

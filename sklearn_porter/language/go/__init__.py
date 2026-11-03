from sklearn_porter.language.LanguageABC import LanguageABC


class Go(LanguageABC):
    KEY = 'go'
    LABEL = 'Go'

    DEPENDENCIES = ['go']
    SUFFIX = 'go'

    # go build -ldflags='-s -w' -o tmp/estimator tmp/estimator.go
    # -s -w omit the symbol table and the debug information, which only
    # debuggers need, so the binaries are about a third smaller (the machine
    # code and the predictions are the same).
    CMD_COMPILE = [
        'go',
        'build',
        '-ldflags=-s -w',
        '-o',
        '{dest_path}',
        '{src_path}',
    ]

    # tmp/estimator <args>
    CMD_EXECUTE = ['{dest_path}']

    # fmt: off
    TEMPLATES = {
        'init':         'var {{ name }} = {{ value }}',

        # if/else condition:
        'if':           'if {{ a }} {{ op }} {{ b }} {',
        'else':         '} else {',
        'endif':        '}',
        # NaN check (e.g. of missing values):
        'isnan':        'math.IsNaN({{ a }})',
        # Positive infinity (e.g. of thresholds):
        'inf':          'math.Inf(1)',
        'nan':          'math.NaN()',

        # Basics:
        'indent':       '\t',
        'join':         '',
        'type':         '{0}',

        # Arrays:
        'in_brackets':  '{{ "{" }}{{ value }}{{ "}" }}',
        # var ages = []int {1, 2}
        'arr[]':        'var {{ name }} = []{{ type }} {{ "{" }}{{ values }}{{ "}" }}',
        'arr[][]':      'var {{ name }} = [][]{{ type }} {{ "{" }}{{ values }}{{ "}" }}',
        'arr[][][]':    'var {{ name }} = [][][]{{ type }} {{ "{" }}{{ values }}{{ "}" }}',

        # Primitive data types:
        'int':          'int',
        'double':       'float64',
        'str':          'string'
    }
    # fmt: on

    @staticmethod
    def str_literal(value: str) -> str:
        # All characters except printable ASCII characters are escaped by
        # their code points:
        chars = [
            char
            if 32 <= ord(char) < 127 and char not in '"\\'
            else f'\\U{ord(char):08x}'
            for char in value
        ]
        return '"' + ''.join(chars) + '"'

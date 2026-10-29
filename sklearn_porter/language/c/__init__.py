from sklearn_porter.language.LanguageABC import LanguageABC


class C(LanguageABC):
    KEY = 'c'
    LABEL = 'C'

    DEPENDENCIES = ['gcc']
    SUFFIX = 'c'

    # gcc tmp/estimator.c -std=c99 -O1 -lm -o tmp/estimator
    # -O1 speeds up the predictions as much as -O2, but compiles large code
    # (e.g. of the `combined` templates) faster. -std=c99 disables the
    # contraction of floating point operations (e.g. to FMA instructions).
    CMD_COMPILE = [
        'gcc',
        '{src_path}',
        '-std=c99',
        '-O1',
        '-lm',
        '-o',
        '{dest_path}',
    ]

    # tmp/estimator <args>
    CMD_EXECUTE = ['{dest_path}']

    # fmt: off
    TEMPLATES = {
        'init':         '{{ type }} {{ name }} = {{ value }};',

        # if/else condition:
        'if':           'if ({{ a }} {{ op }} {{ b }}) {',
        'else':         '} else {',
        'endif':        '}',
        # NaN check (e.g. of missing values):
        'isnan':        'isnan({{ a }})',
        # Positive infinity (e.g. of thresholds):
        'inf':          'INFINITY',
        'nan':          'NAN',

        # Basics:
        'indent':       '    ',
        'join':         '; ',
        'type':         '{{ value }}',

        # Arrays:
        'in_brackets':  '{{ "{" }}{{ value }}{{ "}" }}',
        # const int ages[2] = {1, 2}; (strings: const char *names[2] = ...)
        'arr[]':        '{{ "" if type.startswith("const ") else "const " }}{{ type }} {{ name }}[{{ n }}] = {{ "{" }}{{ values }}{{ "}" }};',
        'arr[][]':      'const {{ type }} {{ name }}[{{ n }}][{{ m }}] = {{ "{" }}{{ values }}{{ "}" }};',
        'arr[][][]':    'const {{ type }} {{ name }}[{{ n }}][{{ m }}][{{ k }}] = {{ "{" }}{{ values }}{{ "}" }};',

        # Primitive data types:
        'int':          'int',
        'double':       'double',
        'str':          'const char *'
    }
    # fmt: on

    @staticmethod
    def str_literal(value: str) -> str:
        # All bytes except printable ASCII characters are escaped as octal
        # numbers, also `?` because of the trigraphs of C99:
        chars = [
            chr(byte)
            if 32 <= byte < 127 and chr(byte) not in '"\\?'
            else f'\\{byte:03o}'
            for byte in value.encode('utf-8')
        ]
        return '"' + ''.join(chars) + '"'

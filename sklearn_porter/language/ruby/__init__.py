import re

from sklearn_porter.language.LanguageABC import LanguageABC, class_members


class Ruby(LanguageABC):
    KEY = 'ruby'
    LABEL = 'Ruby'

    DEPENDENCIES = ['ruby']
    SUFFIX = 'rb'

    CMD_COMPILE = None

    # ruby estimator.rb <args>
    CMD_EXECUTE = ['ruby', '{src_path}']

    # fmt: off
    TEMPLATES = {
        'init':         '{{ name }} = {{ value }}',

        # if/else condition:
        'if':           'if {{ a }} {{ op }} {{ b }}',
        'else':         'else',
        'endif':        'end',
        # NaN check (e.g. of missing values):
        'isnan':        '{{ a }}.nan?',
        # Positive infinity (e.g. of thresholds):
        'inf':          'Float::INFINITY',
        'nan':          'Float::NAN',

        # Basics:
        'indent':       '\t',
        'join':         '',
        'type':         '{{ value }}',

        # Arrays:
        'in_brackets':  '[{{ value }}]',
        'arr[]':        '{{ name }} = [{{ values }}]',
        'arr[][]':      '{{ name }} = [{{ values }}]',
        'arr[][][]':    '{{ name }} = [{{ values }}]',

        # Primitive data types:
        'int':          '',
        'double':       '',
        'str':          ''
    }
    # fmt: on

    @staticmethod
    def str_literal(value: str) -> str:
        # Strings in single quotes contain only the escapes `\\` and `\'`:
        return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"

    @classmethod
    def methods(cls, source: str, class_name: str) -> list[dict]:
        start = re.compile(rf'class {class_name}\b')
        method = re.compile(r'\s*def (self\.)?(predict(?:_index|_proba)?)\b')
        return [
            dict(name=m.group(2), static=bool(m.group(1)))
            for m in map(method.match, class_members(source, start))
            if m
        ]

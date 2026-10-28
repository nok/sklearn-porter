from sklearn_porter.language.LanguageABC import LanguageABC


class Java(LanguageABC):
    KEY = 'java'
    LABEL = 'Java'

    DEPENDENCIES = ['java', 'javac']
    SUFFIX = 'java'

    # javac -cp tmp:gson.jar -d tmp tmp/Estimator.java
    CMD_COMPILE = [
        'javac',
        '-cp',
        '{class_path}',
        '-d',
        '{dest_dir}',
        '{src_path}',
    ]

    # java -cp tmp:gson.jar Estimator <args>
    CMD_EXECUTE = ['java', '-cp', '{class_path}', '{class_name}']

    # fmt: off
    TEMPLATES = {
        'init':         '{{ type }} {{ name }} = {{ value }};',

        # if/else condition:
        'if':           'if ({{ a }} {{ op }} {{ b }}) {',
        'else':         '} else {',
        'endif':        '}',
        # NaN check (e.g. of missing values):
        'isnan':        'Double.isNaN({{ a }})',
        # Positive infinity (e.g. of thresholds):
        'inf':          'Double.POSITIVE_INFINITY',
        'nan':          'Double.NaN',

        # Basics:
        'indent':       '    ',
        'join':         '; ',
        'type':         '{{ value }}',

        # Arrays:
        'in_brackets':  '{{ "{" }}{{ value }}{{ "}" }}',
        'arr[]':        '{{ type }}[] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',
        'arr[][]':      '{{ type }}[][] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',
        'arr[][][]':    '{{ type }}[][][] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',

        # Primitive data types:
        'int':          'int',
        'double':       'double',
        'str':          'String'
    }
    # fmt: on

    GSON_DOWNLOAD_URI = (
        'https://repo1.maven.org/maven2/'
        'com/google/code/gson/gson/2.9.1/gson-2.9.1.jar'
    )

    @staticmethod
    def str_literal(value: str) -> str:
        # Java translates Unicode escapes before it parses the source code,
        # so only characters outside of ASCII are escaped by their UTF-16
        # code units, all other non-printable characters as octal numbers:
        chars = []
        for char in value:
            if 32 <= ord(char) < 127 and char not in '"\\':
                chars.append(char)
            elif ord(char) < 128:
                chars.append(f'\\{ord(char):03o}')
            else:
                units = char.encode('utf-16-be', 'surrogatepass')
                for i in range(0, len(units), 2):
                    unit = int.from_bytes(units[i : i + 2], 'big')
                    chars.append(f'\\u{unit:04x}')
        return '"' + ''.join(chars) + '"'

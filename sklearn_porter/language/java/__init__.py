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
    }
    # fmt: on

    GSON_DOWNLOAD_URI = (
        'https://repo1.maven.org/maven2/'
        'com/google/code/gson/gson/2.9.1/gson-2.9.1.jar'
    )

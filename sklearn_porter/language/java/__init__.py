from sklearn_porter.language.LanguageABC import LanguageABC


class Java(LanguageABC):
    KEY = 'java'
    LABEL = 'Java'

    DEPENDENCIES = ['java', 'javac']
    SUFFIX = 'java'

    # javac {class_path} tmp/Estimator.java
    # class_path = '-cp ./gson.jar'
    CMD_COMPILE = 'javac {class_path} {dest_dir} {src_path}'

    # java {class_path} Estimator <args>
    # class_path = '-cp ./gson.jar:./tmp'
    CMD_EXECUTE = 'java {class_path} {dest_path}'

    # yapf: disable
    TEMPLATES = {
        'init':         '{{ type }} {{ name }} = {{ value }};',

        # if/else condition:
        'if':           'if ({{ a }} {{ op }} {{ b }}) {',
        'else':         '} else {',
        'endif':        '}',

        # Basics:
        'indent':       '    ',
        'join':         '; ',
        'type':         '{{ value }}',

        # Arrays:
        'in_brackets':  '{{ "{" }}{{ value }}{{ "}" }}',
        'arr[]':        '{{ type }}[] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',  # pylint: disable=line-too-long
        'arr[][]':      '{{ type }}[][] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',  # pylint: disable=line-too-long
        'arr[][][]':    '{{ type }}[][][] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',  # pylint: disable=line-too-long

        # Primitive data types:
        'int':          'int',
        'double':       'double',
    }
    # yapf: enable

    GSON_DOWNLOAD_URI = (
        'https://repo1.maven.org/maven2/'
        'com/google/code/gson/gson/2.9.0/gson-2.9.0.jar'
    )

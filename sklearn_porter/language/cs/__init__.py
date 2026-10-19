import json
from functools import cache
from pathlib import Path
from subprocess import CalledProcessError, check_output

from sklearn_porter.language.LanguageABC import LanguageABC


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a version like `10.0.402` or `10.0.100-rc.1.25451.107`."""
    return tuple(int(v) for v in version.split('-')[0].split('.'))


class CSharp(LanguageABC):
    KEY = 'cs'
    LABEL = 'C#'

    DEPENDENCIES = ['dotnet']
    SUFFIX = 'cs'

    # The compiler of the .NET SDK compiles the generated source code
    # directly, which is much faster than a project build (`dotnet build`):
    # dotnet <sdk>/Roslyn/bincore/csc.dll -nologo -langversion:7.3 -optimize+
    #     -out:tmp/Estimator.dll @tmp/Estimator.rsp tmp/Estimator.cs
    CMD_COMPILE = [
        'dotnet',
        '{compiler_path}',
        '-nologo',
        '-langversion:7.3',
        '-optimize+',
        '-out:{dest_path}',
        '@{references_path}',
        '{src_path}',
    ]

    # dotnet tmp/Estimator.dll <args>
    CMD_EXECUTE = ['dotnet', '{dest_path}']

    # fmt: off
    TEMPLATES = {
        'init':         '{{ type }} {{ name }} = {{ value }};',

        # if/else condition:
        'if':           'if ({{ a }} {{ op }} {{ b }}) {',
        'else':         '} else {',
        'endif':        '}',
        # NaN check (e.g. of missing values):
        'isnan':        'double.IsNaN({{ a }})',
        # Positive infinity (e.g. of thresholds):
        'inf':          'double.PositiveInfinity',

        # Basics:
        'indent':       '    ',
        'join':         '; ',
        'type':         '{{ value }}',

        # Arrays (the nested arrays of jagged arrays need their own
        # implicitly typed array creation expressions):
        'in_brackets':  'new[] {{ "{" }}{{ value }}{{ "}" }}',
        'arr[]':        '{{ type }}[] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',
        'arr[][]':      '{{ type }}[][] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',
        'arr[][][]':    '{{ type }}[][][] {{ name }} = {{ "{" }}{{ values }}{{ "}" }};',

        # Primitive data types:
        'int':          'int',
        'double':       'double',
        'str':          'string'
    }
    # fmt: on

    @staticmethod
    def str_literal(value: str) -> str:
        # All characters except printable ASCII characters are escaped by
        # their UTF-16 code units:
        chars = []
        for char in value:
            if 32 <= ord(char) < 127 and char not in '"\\':
                chars.append(char)
            else:
                units = char.encode('utf-16-be', 'surrogatepass')
                for i in range(0, len(units), 2):
                    unit = int.from_bytes(units[i : i + 2], 'big')
                    chars.append(f'\\u{unit:04x}')
        return '"' + ''.join(chars) + '"'

    @staticmethod
    @cache
    def get_sdk() -> tuple[Path, list[Path], str, str]:
        """
        Find the compiler and the reference assemblies of the latest .NET SDK.

        Returns
        -------
        The path of the compiler (`csc.dll`), the paths of the reference
        assemblies, the target framework (e.g. `net10.0`) and the version of
        the runtime (e.g. `10.0.0`).

        Raises
        ------
        RuntimeError
            If no .NET SDK with a compiler and reference assemblies is found.
        """
        try:
            out = check_output(['dotnet', '--list-sdks'], text=True)
        except (OSError, CalledProcessError) as e:
            msg = f'The .NET SDK is not available: {e}'
            raise RuntimeError(msg) from None
        # Each line contains a version and a directory, e.g.
        # `10.0.402 [/usr/share/dotnet/sdk]`:
        sdks = []
        for line in out.splitlines():
            version, _, path = line.strip().partition(' ')
            if version and path:
                sdks.append(
                    (_parse_version(version), Path(path[1:-1]), version)
                )
        for _, sdk_dir, version in sorted(sdks, reverse=True):
            compiler_path = sdk_dir / version / 'Roslyn' / 'bincore' / 'csc.dll'
            packs_dir = sdk_dir.parent / 'packs' / 'Microsoft.NETCore.App.Ref'
            if not compiler_path.exists() or not packs_dir.is_dir():
                continue
            refs = [p for p in packs_dir.iterdir() if p.name[:1].isdigit()]
            refs = sorted(refs, key=lambda p: _parse_version(p.name))
            for ref_dir in reversed(refs):
                for framework_dir in (ref_dir / 'ref').glob('net*'):
                    references = sorted(framework_dir.glob('*.dll'))
                    if references:
                        # The runtime rolls forward to its latest patch:
                        major, minor = _parse_version(ref_dir.name)[:2]
                        return (
                            compiler_path,
                            references,
                            framework_dir.name,
                            f'{major}.{minor}.0',
                        )
        msg = 'No .NET SDK with a C# compiler and reference assemblies found.'
        raise RuntimeError(msg)

    @classmethod
    def prepare_compilation(cls, src_path: Path) -> tuple[dict, list[Path]]:
        """
        Write the files to compile and execute a generated source file.

        The compiler gets the reference assemblies of the .NET SDK from a
        response file and the compiled program needs a runtime configuration
        to be executed by `dotnet`. Both files are written next to the source
        file.

        Parameters
        ----------
        src_path : Path
            The path of the generated source file.

        Returns
        -------
        The arguments of the commands `CMD_COMPILE` and `CMD_EXECUTE` and the
        paths of the created files (incl. the compiled program).
        """
        compiler_path, references, framework, version = cls.get_sdk()
        dest_path = src_path.with_suffix('.dll')
        references_path = src_path.with_suffix('.rsp')
        references_path.write_text(
            ''.join(f'-r:"{path}"\n' for path in references),
            encoding='utf-8',
        )
        config = dict(
            runtimeOptions=dict(
                tfm=framework,
                framework=dict(name='Microsoft.NETCore.App', version=version),
            )
        )
        config_path = src_path.with_suffix('.runtimeconfig.json')
        config_path.write_text(json.dumps(config), encoding='utf-8')
        cmd_args = dict(
            compiler_path=str(compiler_path),
            references_path=str(references_path),
            src_path=str(src_path),
            dest_path=str(dest_path),
        )
        return cmd_args, [dest_path, references_path, config_path]

"""
Test the ported estimators and transformers as library in own programs.

The program of each language is the usage in the header of the ported source
code (e.g. the factory `create` of the template `attached` and `load` of the
template `exported`), so the test checks the documented usage. The ported
JavaScript as module is tested by `test_js_module*`. The pipelines (the class
of the pipeline, in C the functions with the prefix `pipeline_`) apply their
transformers before the predictions of the final estimator.
"""

import json
import os
import re
import warnings
from pathlib import Path
from subprocess import run
from typing import Any

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes, load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn_porter import Porter, can
from sklearn_porter.language.cs import CSharp
from tests.commons import CLASSIFIERS, OUTLIER_DETECTORS, REGRESSORS
from tests.utils import fs_mkdir

LANGUAGES = ['c', 'cs', 'go', 'java', 'php', 'ruby']

DOCUMENTS = [
    'the quick brown fox',
    'the lazy dog and the quick cat',
    'a brown dog',
]


def _iris() -> tuple[np.ndarray, np.ndarray]:
    return load_iris(return_X_y=True)


def _column_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ('scaler', StandardScaler(), [0, 1]),
            ('steps', make_pipeline(StandardScaler(), MinMaxScaler()), [2, 3]),
        ]
    )


TRANSFORMERS = {
    'StandardScaler': lambda: (StandardScaler().fit(_iris()[0]), None),
    'ColumnTransformer': lambda: (_column_transformer().fit(_iris()[0]), None),
    'FeatureUnion': lambda: (
        FeatureUnion([('a', StandardScaler()), ('b', MinMaxScaler())]).fit(
            _iris()[0]
        ),
        None,
    ),
    'SelectKBest': lambda: (SelectKBest(k=2).fit(*_iris()), None),
    'OneHotEncoder': lambda: (
        OneHotEncoder(handle_unknown='ignore').fit(np.round(_iris()[0])),
        np.round(_iris()[0][::25]).tolist(),
    ),
    'CountVectorizer': lambda: (
        CountVectorizer().fit(DOCUMENTS),
        DOCUMENTS + ['the unknown dog'],
    ),
}

ESTIMATORS = {c.name: c for c in CLASSIFIERS + REGRESSORS + OUTLIER_DETECTORS}

# Pipelines of two transformers and each estimator:
PIPELINES = [f'Pipeline({name})' for name in ESTIMATORS]
# A pipeline with a composite transformer:
COMPOSITE_PIPELINE = 'Pipeline(ColumnTransformer, DecisionTreeClassifier)'

CANDIDATES = (
    list(ESTIMATORS) + list(TRANSFORMERS) + PIPELINES + [COMPOSITE_PIPELINE]
)


def _create(name: str) -> tuple[Any, list, bool]:
    """Fit a candidate, return it with the samples and whether it is a
    transformer."""
    if name in TRANSFORMERS:
        est, samples = TRANSFORMERS[name]()
        if samples is None:
            samples = _iris()[0][::25].tolist()
        return est, samples, True
    if name == COMPOSITE_PIPELINE:
        x, y = _iris()
        est = make_pipeline(_column_transformer(), DecisionTreeClassifier())
        return est.fit(x, y), x[::25].tolist(), False
    final_name = name.removeprefix('Pipeline(').removesuffix(')')
    candidate = ESTIMATORS[final_name]
    est = candidate.create()
    # Few trees are enough for the generated code:
    for param in ('n_estimators', 'max_iter'):
        if param in est.get_params() and 'MLP' not in name:
            est.set_params(**{param: 2})
    if name in PIPELINES:
        est = make_pipeline(StandardScaler(), MinMaxScaler(), est)
    x, y = _iris()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # e.g. of the convergence
        if candidate in CLASSIFIERS:
            est.fit(x, y)
        elif candidate in REGRESSORS:
            x, y = load_diabetes(return_X_y=True)
            x = x[:, :4]
            est.fit(x, y)
        else:
            est.fit(x)
    return est, x[::25].tolist(), False


def _usage(code: str) -> list[str]:
    """The lines of the usage of the class in own code (in the header)."""
    lines = code.splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if re.search(r'Use (the class|it) in your code', line)
    )
    usage = []
    for line in lines[start + 1 :]:
        match = re.match(r'^(?: \*|#)?(?: {8,}|\t{2,})(\S.*)$', line)
        if not match:
            break
        usage.append(match.group(1))
    assert usage
    return usage


def _literal(lang: str, samples: list) -> str:
    """The samples (numbers or documents) as literal of the language."""
    if isinstance(samples[0], str):
        items = ', '.join(json.dumps(s) for s in samples)
        return {
            'c': f'{{{items}}}',
            'cs': f'new string[] {{ {items} }}',
            'go': f'[]string{{{items}}}',
            'java': f'new String[] {{ {items} }}',
            'php': f'[{items}]',
            'ruby': f'[{items}]',
        }[lang]
    rows = [', '.join(repr(float(v)) for v in s) for s in samples]
    return {
        'c': '{' + ', '.join(f'{{{r}}}' for r in rows) + '}',
        'cs': 'new double[][] { '
        + ', '.join(f'new double[] {{ {r} }}' for r in rows)
        + ' }',
        'go': '[][]float64{' + ', '.join(f'{{{r}}}' for r in rows) + '}',
        'java': 'new double[][] { '
        + ', '.join(f'{{ {r} }}' for r in rows)
        + ' }',
        'php': '[' + ', '.join(f'[{r}]' for r in rows) + ']',
        'ruby': '[' + ', '.join(f'[{r}]' for r in rows) + ']',
    }[lang]


# The placeholders of the features or the document in the usage:
ARGUMENTS = [
    r'new double\[\] \{ feature_1, \.\.\., feature_\d+ \}',
    r'\[\]float64\{feature_1, \.\.\., feature_\d+\}',
    r'\[feature_1, \.\.\., feature_\d+\]',
    r'"document"',
    r"'document'",
]


def _program(
    lang: str,
    usage: list[str],
    samples: list,
    is_transformer: bool,
    data_path: str | None,
    pointer: bool = False,
) -> str:
    """The source code of a program with the usage of the header."""
    if data_path:
        usage = [
            re.sub(r'"\w+\.json"', json.dumps(data_path), line)
            for line in usage
        ]
        usage = [
            re.sub(r"'\w+\.json'", f"'{data_path}'", line) for line in usage
        ]
    *decls, call = usage
    var = '$s' if lang == 'php' else 's'
    if lang == 'c':
        call = re.sub(r'\b(features|document)\)', 'samples[i])', call)
        var = 'samples[i]'
    else:
        for pattern in ARGUMENTS:
            call = re.sub(pattern, lambda _: var, call)
    assert not re.search(r'feature_|document', call), call
    call = call.rstrip(';')
    lit = _literal(lang, samples)
    text = isinstance(samples[0], str)
    if lang == 'java':
        kind = 'String' if text else 'double[]'
        body = f'print({call});'
        decls = '\n        '.join(decls)
        return (
            'public class Host {\n'
            '    static void print(Object v) { System.out.println(v); }\n'
            '    static void print(double[] v) {\n'
            '        for (double e : v) System.out.println(e);\n'
            '    }\n'
            '    public static void main(String[] args) throws Exception {\n'
            f'        {decls}\n'
            f'        for ({kind} s : {lit}) {{\n'
            f'            {body}\n'
            '        }\n'
            '    }\n'
            '}\n'
        )
    if lang == 'cs':
        kind = 'string' if text else 'double[]'
        body = f'Print({call});'
        decls = '\n        '.join(decls)
        return (
            'using System;\nusing System.Globalization;\n\n'
            'public static class Host {\n'
            '    static void Print(object v) {\n'
            '        Console.WriteLine(Convert.ToString(v, '
            'CultureInfo.InvariantCulture));\n'
            '    }\n'
            '    static void Print(double[] v) {\n'
            '        foreach (double e in v) Print(e);\n'
            '    }\n'
            '    public static void Main(string[] args) {\n'
            f'        {decls}\n'
            f'        foreach ({kind} s in {lit}) {{\n'
            f'            {body}\n'
            '        }\n'
            '    }\n'
            '}\n'
        )
    if lang == 'go':
        lines = []
        for decl in decls:
            lines.append(decl)
            if ', err :=' in decl:
                lines.append('if err != nil {\n\t\tpanic(err)\n\t}')
        if is_transformer:
            body = f'for _, v := range {call} {{\n\t\t\tfmt.Println(v)\n\t\t}}'
        else:
            body = f'fmt.Println({call})'
        decls = '\n\t'.join(lines)
        return (
            'package main\n\nimport "fmt"\n\n'
            'func main() {\n'
            f'\t{decls}\n'
            f'\tfor _, s := range {lit} {{\n'
            f'\t\t{body}\n'
            '\t}\n'
            '}\n'
        )
    if lang == 'php':
        if is_transformer:
            body = (
                f'foreach ({call} as $v) {{\n'
                '        echo json_encode($v), "\\n";\n'
                '    }'
            )
        else:
            body = f'echo json_encode({call}), "\\n";'
        decls = '\n'.join(decls)
        return f'<?php\n{decls}\nforeach ({lit} as $s) {{\n    {body}\n}}\n'
    if lang == 'ruby':
        if is_transformer:
            body = f'{call}.each {{ |v| puts v }}'
        else:
            body = f'puts {call}'
        decls = '\n'.join(decls)
        return f'{decls}\n{lit}.each do |s|\n  {body}\nend\n'
    # C:
    head = '\n'.join(d for d in decls if d.startswith('#'))
    decls = '\n    '.join(d for d in decls if not d.startswith('#'))
    n = len(samples)
    if text:
        array = f'const char *samples[{n}] = {lit};'
    else:
        array = f'double samples[{n}][N_FEATURES] = {lit};'
    if is_transformer:
        body = (
            'double values[N_FEATURES_OUT];\n'
            f'        {call};\n'
            '        for (int j = 0; j < N_FEATURES_OUT; j++) {\n'
            '            printf("%.17g\\n", values[j]);\n'
            '        }'
        )
    elif 'values' in call:  # e.g. the values of all outputs
        body = (
            'double values[N_OUTPUTS];\n'
            f'        {call};\n'
            '        for (int j = 0; j < N_OUTPUTS; j++) {\n'
            '            printf("%.17g\\n", values[j]);\n'
            '        }'
        )
    elif pointer:  # the values of the leaf
        body = f'printf("%.17g\\n", *{call});'
    else:
        body = f'printf("%.17g\\n", (double) {call});'
    return (
        f'{head}\n\n'
        'int main(void) {\n'
        f'    {decls}\n'
        f'    {array}\n'
        f'    for (int i = 0; i < {n}; i++) {{\n'
        f'        {body}\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )


def _run(cmd: list[str], cwd: Path) -> str:
    proc = run(cmd, cwd=cwd, capture_output=True, encoding='utf-8')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not proc.stderr, proc.stderr
    # e.g. the warnings and deprecations of PHP (see `strict_php`):
    assert not any(w in proc.stdout for w in ('Warning:', 'Deprecated:'))
    return proc.stdout


def _execute(lang: str, tmp_dir: Path, src_path: Path, program: str) -> str:
    """Compile (if necessary) and execute the program with the class."""
    if lang == 'java':
        (tmp_dir / 'Host.java').write_text(program)
        class_path = os.pathsep.join(
            [str(tmp_dir), os.environ['SKLEARN_PORTER_GSON_PATH']]
        )
        cmd = ['javac', '-cp', class_path, '-d', str(tmp_dir)]
        cmd += [str(src_path), 'Host.java']
        proc = run(cmd, cwd=tmp_dir, capture_output=True, encoding='utf-8')
        if 'code too large' in proc.stdout + proc.stderr:
            pytest.skip('Skip too large code (Java)')
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return _run(['java', '-cp', class_path, 'Host'], tmp_dir)
    if lang == 'cs':
        (tmp_dir / 'Host.cs').write_text(program)
        args = CSharp.prepare_compilation(src_path)[0]
        cmd = [arg.format(**args) for arg in CSharp.CMD_COMPILE or []]
        cmd += ['Host.cs', '-define:SKLEARN_PORTER_NO_MAIN']
        _run(cmd, tmp_dir)
        return _run([a.format(**args) for a in CSharp.CMD_EXECUTE], tmp_dir)
    if lang == 'go':
        # Like the usage in the header: remove `main` of the ported code:
        code = src_path.read_text()
        code = re.sub(r'\nfunc main\(\) \{\n.*?\n\}\n', '\n', code, flags=re.S)
        assert 'func main()' not in code
        src_path.write_text(code)
        (tmp_dir / 'host.go').write_text(program)
        cmd = ['go', 'build', '-o', 'host', src_path.name, 'host.go']
        _run(cmd, tmp_dir)
        return _run(['./host'], tmp_dir)
    if lang == 'php':
        (tmp_dir / 'host.php').write_text(program)
        return _run(['php', '-f', 'host.php'], tmp_dir)
    if lang == 'ruby':
        (tmp_dir / 'host.rb').write_text(program)
        return _run(['ruby', 'host.rb'], tmp_dir)
    (tmp_dir / 'host.c').write_text(program)
    _run(['gcc', 'host.c', '-std=c99', '-O1', '-lm', '-o', 'host'], tmp_dir)
    return _run(['./host'], tmp_dir)


@pytest.mark.regression
@pytest.mark.parametrize('template', ['attached', 'exported', 'combined'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize('name', CANDIDATES)
def test_library_usage(
    tmp_root_dir: Path, name: str, language: str, template: str
):
    """Use the ported class in an own program like the usage in the header."""
    est, samples, is_transformer = _create(name)
    method = 'transform' if is_transformer else 'predict'
    if not can(est, language, template, method):
        pytest.skip('Skip unsupported estimator/language/template combination')
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'library_usage'),
            ('name', name),
            ('language', language),
            ('template', template),
        ],
    )
    out = Porter(est, language=language, template=template).save(
        directory=tmp_dir
    )
    if isinstance(out, tuple):  # the source code and the model data
        src_path, data_path = Path(out[0]), str(out[1])
    else:
        src_path, data_path = Path(out), None
    code = src_path.read_text()
    program = _program(
        language,
        _usage(code),
        samples,
        is_transformer,
        data_path,
        pointer=bool(re.search(r'double \*predict\(', code)),
    )
    stdout = _execute(language, tmp_dir, src_path, program)
    # The values (e.g. of arrays like `[1.5, 2.5]` of several outputs):
    values = [float(v) for v in re.split(r'[\s\[\],]+', stdout) if v]
    expected = (
        est.transform(samples) if is_transformer else est.predict(samples)
    )
    if hasattr(expected, 'toarray'):
        expected = expected.toarray()
    expected = np.asarray(expected, dtype=float).ravel()
    final_name = name.removeprefix('Pipeline(').removesuffix(')')
    final_name = final_name.split(', ')[-1]
    if ESTIMATORS.get(final_name) in CLASSIFIERS + OUTLIER_DETECTORS:
        np.testing.assert_array_equal(values, expected)
    else:
        np.testing.assert_allclose(values, expected, rtol=1e-9)

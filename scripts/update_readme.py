"""
Generate the support table in README.md from the `SUPPORT` declarations.

Usage:
    uv run python scripts/update_readme.py          # update README.md
    uv run python scripts/update_readme.py --check  # fail if outdated
"""

import sys
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

from sklearn_porter import Estimator
from sklearn_porter import enums as enum
from sklearn_porter.estimator.EstimatorBase import EstimatorBase

README = Path(__file__).resolve().parent.parent / 'README.md'
START = '<!-- support-table:start -->'
END = '<!-- support-table:end -->'

SKLEARN_URL = 'https://scikit-learn.org/stable/modules/generated/'
LANGUAGES = list(enum.Language)
TEMPLATES = [
    (enum.Template.ATTACHED, 'ᴀ'),
    (enum.Template.EXPORTED, 'ᴇ'),
    (enum.Template.COMBINED, 'ᴄ'),
]
N_COLS = len(LANGUAGES) * len(TEMPLATES)


def _load(estimator: type) -> type[EstimatorBase]:
    """Load the sklearn-porter class of a scikit-learn estimator."""
    name = estimator.__name__
    module = import_module(f'sklearn_porter.estimator.{name}')
    return getattr(module, name)


def _row(clazz: type[EstimatorBase]) -> list[str]:
    url = clazz.SKLEARN_URL or ''
    label = url.removeprefix('sklearn.').removesuffix('.html')
    lines = [
        '  <tr align="center">',
        '    <td align="left">',
        f'      <a href="{SKLEARN_URL}{url}">{label}</a>',
        '    </td>',
    ]
    for language in LANGUAGES:
        support = clazz.SUPPORT.get(language, {})
        for template, _ in TEMPLATES:
            methods = support.get(template, set())
            if enum.Method.PREDICT_PROBA in methods:
                cell = '✓ᴾ'
            elif enum.Method.PREDICT in methods:
                cell = '✓'
            else:
                cell = '×'
            lines.append(f'    <td>{cell}</td>')
    lines.append('  </tr>')
    return lines


def render_table() -> str:
    """
    Render the support table of all estimators as HTML.

    Returns
    -------
    The support table, classifiers first, each group sorted by the
    module path of scikit-learn.
    """
    lines = [
        '<table>',
        '  <tr>',
        '    <th rowspan="2"></th>',
        f'    <th colspan="{N_COLS}">Programming language</th>',
        '  </tr>',
        '  <tr align="center">',
    ]
    for language in LANGUAGES:
        label = language.value.LABEL
        lines.append(f'    <th colspan="{len(TEMPLATES)}">{label}</th>')
    lines.append('  </tr>')
    for group in (Estimator.classifiers(), Estimator.regressors()):
        classes = sorted(map(_load, group), key=lambda c: c.SKLEARN_URL or '')
        for clazz in classes:
            lines.extend(_row(clazz))
    lines += ['  <tr align="center">', '    <td rowspan="2"></td>']
    for _ in LANGUAGES:
        lines.extend(f'    <td>{symbol}</td>' for _, symbol in TEMPLATES)
    lines += [
        '  </tr>',
        '  <tr>',
        f'    <th colspan="{N_COLS}">Template</th>',
        '  </tr>',
        '</table>',
    ]
    return '\n'.join(lines)


def update(text: str) -> str:
    """
    Replace the support table between the markers.

    Parameters
    ----------
    text : str
        The content of README.md.

    Returns
    -------
    The content of README.md with the current support table.
    """
    head, sep, rest = text.partition(START)
    _, sep_end, tail = rest.partition(END)
    if not sep or not sep_end:
        msg = f'The markers `{START}` and `{END}` are missing in README.md.'
        raise ValueError(msg)
    return f'{head}{START}\n{render_table()}\n{END}{tail}'


def main() -> int:
    parser = ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check whether README.md is up to date.',
    )
    args = parser.parse_args()
    text = README.read_text(encoding='utf-8')
    new_text = update(text)
    if args.check:
        if new_text != text:
            print('README.md is outdated, run `make readme`.')
            return 1
        print('README.md is up to date.')
        return 0
    README.write_text(new_text, encoding='utf-8')
    print('Updated README.md.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

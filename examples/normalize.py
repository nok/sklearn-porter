"""
Normalize the executed notebooks (see `make examples`), so that they only
change with their content: stable IDs of the cells and no execution times.
"""

import sys

import nbformat

for path in sys.argv[1:]:
    notebook = nbformat.read(path, as_version=4)
    for i, cell in enumerate(notebook.cells):
        cell.id = f'cell-{i}'
        cell.metadata.pop('execution', None)
    nbformat.write(notebook, path)

#!/usr/bin/env bash

source activate sklearn-porter
python -m $(python -c 'import sys; print("http.server" if sys.version_info[:2] > (2,7) else "SimpleHTTPServer")') 8080 & serve_pid=$!
python -m unittest discover -vp '*Test.py'
kill $serve_pid
source deactivate

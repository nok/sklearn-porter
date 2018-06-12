#!/usr/bin/env bash

# Activate the relevant environment:
source activate sklearn-porter

# Start local server which is required for the javascript tests:
if [[ $(python -c "import sys; print(sys.version_info[:1][0]);") == "2" ]]; then
  python -m SimpleHTTPServer 8080 &>/dev/null & serve_pid=$!;
else
  python -m http.server 8080 &>/dev/null & serve_pid=$!;
fi

# Run all tests:
python -m unittest discover -vp '*Test.py'

# N_RANDOM_FEATURE_SETS=15 N_EXISTING_FEATURE_SETS=30 python -m unittest discover -vp '*Test.py'
# python -m unittest discover -vp 'RandomForest*Test.py'
# python -m unittest discover -vp '*JavaTest.py'

# Close the previous started server:
kill $serve_pid

# Deactivate the previous activated environment:
source deactivate &>/dev/null

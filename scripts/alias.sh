
function porter() {
  if [[ "$(pip list --format=columns | grep sklearn-porter)" ]]
  then
    local CMDS="from distutils.sysconfig import get_python_lib;\
                print(get_python_lib())"
    local LIBS=$(python -c "$CMDS")
    python $LIBS/sklearn_porter "$@"
  else
     echo "Error: The module 'sklearn-porter' could not be found. Is 'sklearn-porter' installed and the right environment active?"
  fi
}

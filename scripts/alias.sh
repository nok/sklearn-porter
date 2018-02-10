
function porter() {
  if [[ "$(pip list --format=columns | grep sklearn-porter)" ]]
  then
    local CMDS="from distutils.sysconfig import get_python_lib;\
                print(get_python_lib())"
    local LIBS=$(python -c "$CMDS")
    python $LIBS/sklearn_porter "$@"
  else
     echo "Cammond not found."
  fi
}

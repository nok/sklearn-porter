#!/usr/bin/env bash

read -r -p "Install function 'porter' to ~/.bash_profile? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
    cat $SCRIPTPATH/function.sh >> ~/.bash_profile
    source ~/.bash_profile
fi

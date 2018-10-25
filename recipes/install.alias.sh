#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cat $SCRIPTPATH/alias.sh >> ~/.bash_profile
source ~/.bash_profile

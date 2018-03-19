#!/bin/bash
# courtesy of: http://redsymbol.net/articles/unofficial-bash-strict-mode/
# (helps with debugging)
# set -e: immediately exit if we find a non zero
# set -u: undefined references cause errors
# set -o: single error causes full pipeline failure.
set -euo pipefail
IFS=$'\n\t'
# datestring, used in many different places...
dateStr=`date +%Y-%m-%d:%H:%M:%S`

# Description:

# Arguments:
# This file runs the full analysis pipeline for generating figure-ready data.
#### Arg 1: Description
data_base="../../Data/FECs180307/"
bash run_on_all_dirs.sh ../Analysis/main/main.sh "$data_base"





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
#### Arg 1: Description

# Returns:

# note: following line works in one directory above Data"
#  rsync.exe -Rv ./Data/rationale/N2/cache_0_binding/binding.pkl ../../../Dropbox/scratch/


input=${1:-"../../../Data/FECs180307/"}
output=${2:-"~/Dropbox/scratch/"}
cd "$input"
for f in `find ./ -name "*energy.pkl"`
    do 
        rsync.exe -Rrv $f "${output}"
    done

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
data_dir=${1:-../../../Data/BR+Retinal/170321FEC/}
files=`find .. -name "main*.py" | sort`
for f in $files
    do
        file_base=`basename $f`
        dir_name=`dirname $f`
        cd $dir_name
        echo "Running $file_base on $data_dir"
        python "$file_base" --base "$data_dir" || echo "Couldn't run $file_base"
        cd -
    done

# Returns:




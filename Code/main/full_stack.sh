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


function full_stack(){
    dir="$1"
	skip_process=$2
	process_input=$3
	if [[ "$skip_process" = 1 ]]; then
		echo "Skipping processing"
	else
		bash process.sh "$dir" "$process_input"
	fi
    bash generate_landscapes.sh "$dir"
}

full_stack "$1" $2 $3




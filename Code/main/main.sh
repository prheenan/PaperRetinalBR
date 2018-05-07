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

data_base="../../Data/FECs180307/"
# remove all the landscape caches, since we may have changed the blacklist
find "$data_base"  -path "*landscape_cache*" -type f -exec rm -f {} \;
skip_reading=${1:-1}
skip_process=${2:-1}
bash full_stack.sh "${data_base}BR+Retinal/" $skip_process $skip_reading
bash full_stack.sh "${data_base}BR-Retinal/" $skip_process $skip_reading
bash analysis.sh "${data_base}"





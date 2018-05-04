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
skip_reading=1
bash full_stack.sh "${data_base}BR+Retinal/" $skip_reading
bash full_stack.sh "${data_base}BR-Retinal/" $skip_reading
bash analysis.sh "${data_base}"





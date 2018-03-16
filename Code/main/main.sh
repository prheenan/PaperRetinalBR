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

function cd_prh(){
    cd "$1" > /dev/null
}

function run_on_all_dirs(){
    bash_file=$1
    dir_to_search=$2
    sub_files=``
    files=`find "$dir_to_search" -mindepth 1 -maxdepth 1 -type d`    
    for g in $files
        do
            # determine where the directory we send as an argument to the file
            cd_prh $g
            abs_f=`pwd`
            cd_prh -
            # determine where the bash file lives
            dir_to_run=`dirname $bash_file`
            file_to_run=`basename $bash_file`
            # run the bash file where it lives
            cd_prh "$dir_to_run"
            bash $file_to_run "$abs_f/"
            # go back
            cd -
        done
}

# Description:
function process(){
    dir="$1"
    cd_prh "$dir"
    # determine the absolute directory
    abs_dir=`pwd`
    cd_prh -
    # find all the subdirectories (different velocities
    files=`find "$abs_dir" -mindepth 1 -maxdepth 1 -type d`
    # process everything in the subdirectories
    for f in $files
        do
            print("===== Processing directory $f ====")
            run_on_all_dirs ../Processing/main/main.sh "$f/"
        done
}
 

# Arguments:
# This file runs the full analysis pipeline for generating figure-ready data.
#### Arg 1: Description
data_base="../../Data/"
process "${data_base}BR+Retinal/"
# Returns:




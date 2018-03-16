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

function invalid_directory(){
    if [[ "$1" = *"cache"* ]]; then
        return 0;
    else
        return 1;
    fi
}

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
            if invalid_directory "$g"; then
                continue
            fi
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

function run_recursive(){
    descr="$1"
    dir="$2"
    bash_file="$3"
    cd_prh "$dir"
    # determine the absolute directory
    abs_dir=`pwd`
    cd_prh -
    # find all the subdirectories (different velocities
    files=`find "$abs_dir" -mindepth 1 -maxdepth 1 -type d`
    # process everything in the subdirectories
    for f in $files
        do
            if invalid_directory "$f"; then
                continue
            fi
           echo "===== $descr $f ====" 
           run_on_all_dirs "$bash_file" "$f/"
        done  
}

function process(){
    run_recursive "Processing for" "$1" ../Processing/main/main.sh
}

function generate_landscapes(){
    run_recursive "Generating landscapes for" "$1" ../Landscapes/main/main.sh
}

function full_stack(){
    dir="$1"
    process "$dir"
    generate_landscapes "$dir"
}
 
# Description:

# Arguments:
# This file runs the full analysis pipeline for generating figure-ready data.
#### Arg 1: Description
data_base="../../Data/"
full_stack "${data_base}BR-Retinal/"
full_stack "${data_base}BR+Retinal/"
# Returns:




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
    # Returns: true iff the directory isn't a cache directory
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
    # Runs the first arg (a .sh file) on all subdirectories in the second arg
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
            bash $file_to_run "$abs_f/" "${@:3}"
            # go back
            cd -
        done
}

# Args:
#   1: what to run
#   2: run on each directory under this directory
run_on_all_dirs "$1" "$2" "${@:3}"





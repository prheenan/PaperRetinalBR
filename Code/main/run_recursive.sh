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
           bash run_on_all_dirs.sh "$bash_file" "$f/"
        done  
}

run_recursive "$@"
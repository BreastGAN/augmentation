#!/bin/bash

function scripttest {
    echo -ne "\nPython cleaning test... "
    stagedfiles=$(git diff --cached --name-only | grep '.py$')
    if [ -z "$stagedfiles" ]; then
        echo -e "PASSED"
        return 0
    fi

    unformatted=$(yapf --diff $stagedfiles | grep "(reformatted)" | awk '{print $2}')
    if [ -z "$unformatted" ]; then
        echo -e "PASSED"
        return 0
    fi

    echo "The following files are not formatted properly:"
    for fn in $unformatted; do
        echo "    $fn"
    done

    echo -e "\nPython files must be formatted with YAPF. Please run:"
    echo "    ./setup/clean.sh"
    return 1
}

function notebooktest {
    echo -ne "\nNotebook cleaning test... "
    stagedfiles=$(git diff --cached --name-only | grep '.ipynb$')
    if [ -z "$stagedfiles" ]; then
        echo -e "PASSED"
        return 0
    fi

    unformatted=$(python resources/yapf_nbformat/yapf_nbformat.py --dry_run $stagedfiles | grep "(reformatted)" | awk '{print $2}')
    if [ -z "$unformatted" ]; then
        echo -e "PASSED"
        return 0
    fi

    echo "The following files are not formatted properly:"
    for fn in $unformatted; do
        echo "    $fn"
    done

    echo -e "\nNotebook files must be formatted with a custom nbformat-YAPF script. Please run:"
    echo "    ./setup/clean.sh"
    return 1
}

# Redirect output to stderr.
exec 1>&2

echo "Running pre-commit hooks..."

./setup/tests.sh
RESULT=$?
[ $RESULT -ne 0 ] && exit 1

scripttest || exit 1
notebooktest || exit 1

echo -e "\nPre-commit hooks PASSED"

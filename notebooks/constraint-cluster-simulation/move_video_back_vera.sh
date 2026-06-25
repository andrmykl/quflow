#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 -n N"
    exit 1
}

N=""

while getopts ":n:" opt; do
    case "$opt" in
        n) N="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$N" ]]; then
    usage
fi

REMOTE_FILE="mysim_N_${N}.mp4"
REMOTE_HOST="vera1"
REMOTE_USER="andmyk"

# Optional: local destination (current directory)
LOCAL_DIR="."

echo "Copying ${REMOTE_FILE} from ${REMOTE_HOST}..."

scp "${REMOTE_HOST}:~/${REMOTE_FILE}" "${LOCAL_DIR}/"

echo "Done: ${LOCAL_DIR}/${REMOTE_FILE}"

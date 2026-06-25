#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 -n N -t SIMTIME -c CORES"
    exit 1
}

N=""
SIMTIME=""
CORES=""

while getopts ":n:t:c:" opt; do
    case "$opt" in
        n) N="$OPTARG" ;;
        t) SIMTIME="$OPTARG" ;;
        c) CORES="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$N" || -z "$SIMTIME" || -z "$CORES" ]]; then
    usage
fi

HDF5_FILE="mysim_N_${N}.hdf5"
PY_FILE="mysim_runfile.py"
SBATCH_FILE="run_vera.sh"

for f in "$HDF5_FILE" "$PY_FILE" "$SBATCH_FILE"; do
    [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 1; }
done

scp "$HDF5_FILE" "$PY_FILE" "$SBATCH_FILE" vera1:~/

ssh vera1 "
    cd ~ &&
    sbatch -c ${CORES} ${SBATCH_FILE} ${PY_FILE} --simtime ${SIMTIME} -f ${HDF5_FILE}
"

sleep 5

ssh vera1 "squeue -u andmyk"

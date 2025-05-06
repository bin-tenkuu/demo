#!/bin/bash
cd "$(dirname "$0")"

DIRS=(
  dd
  sd
  wd
)

set -exo pipefail

for i in "${DIRS[@]}" ; do
  cd "$i" || exit 1
  ./run.sh
  cd ..
done

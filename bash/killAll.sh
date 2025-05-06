#!/bin/bash
cd "$(dirname "$0")"

PORTS=(
  8081
  8082
  8089
)

set -exo pipefail

for i in "${PORTS[@]}" ; do
  PID=$(lsof -t -i:"$i")
  if [ -n "$PID" ]; then
    ps "$PID"
    kill -9 "$PID"
  fi
done

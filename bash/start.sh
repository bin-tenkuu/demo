#!/bin/bash
cd "$(dirname "$0")"
cmd="sleep 10000"
stdout="nohup.out"
pidFile="pid"

use_nohup=0
while [ $# != 0 ]; do
  case $1 in
  fg)
    use_nohup=0
    ;;
  nohup|bg)
    use_nohup=1
    ;;
  tail)
    exec tail -f "$stdout"
    ;;
  kill)
    kill "$(cat "$pidFile")"
    ;;
  pid)
    exec cat "$pidFile"
    ;;
  ps)
    exec ps -ef "$(cat "$pidFile")"
    ;;
  *)
    cmd="$cmd \"$1\""
    ;;
  esac
  shift
done

set -exo pipefail
if [ "$use_nohup" == 1 ]; then
  eval "$cmd" > stdout.log 2>&1 &
  disown "$(jobs -p)"
  tail -f stdout.log
else
  eval "$cmd"
fi

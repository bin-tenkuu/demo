#!/bin/bash
cd "$(dirname "$0")"
cmd="sleep 10000"
stdout="nohup.out"
pidFile="pid"

use_nohup=1
while [ $# != 0 ]; do
  case $1 in
  fg)
    use_nohup=0
    ;;
  nohup|bg)
    use_nohup=1
    ;;
  tail|log)
    exec tail -f -n1000 "$stdout"
    ;;
  kill|stop)
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

pid=$(cat "$pidFile")
if [ -n "$pid" ]; then
  if ps -p "$pid" > /dev/null; then
    exec tail -f "$stdout"
  fi
fi

set -exo pipefail
if [ "$use_nohup" == 1 ]; then
  eval "$cmd > $stdout 2>&1 &"
  disown "$(jobs -p)"
  tail -f "$stdout"
else
  exec "$cmd"
fi

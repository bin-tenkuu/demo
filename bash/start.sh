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
    exec kill "$(cat "$pidFile")"
    ;;
  pid)
    exec cat "$pidFile"
    ;;
  ps)
    exec ps "$(cat "$pidFile")"
    ;;
  *)
    echo "Unknown command: $1"
    exit 1
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
  # shellcheck disable=SC2086
  nohup $cmd > $stdout 2>&1 &
  pid=$!
  echo $pid > $pidFile
  tail -f $stdout
else
  exec $cmd
fi

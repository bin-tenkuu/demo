#!/bin/bash

cmd="java"

use_nohup=0
while [ $# != 0 ]; do
  case $1 in
  nohup)
    use_nohup=1
    cmd="nohup $cmd"
    ;;
  *)
    cmd="$cmd \"$1\""
  esac
  shift
done

if [ "$use_nohup" == 1 ]; then
  cmd="nohup $cmd > /dev/null 2>&1 &"
fi

set -exo pipefail
eval "$cmd"

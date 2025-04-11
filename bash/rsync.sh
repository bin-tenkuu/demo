#!/bin/bash
set -exo pipefail

HOST="127.0.0.1"
USERNAME="bin-"
PASSWORD="12345678"
REMOTE_DIR="/home/bin-/backup"
LOCAL_DIR="/home/bin-/"

expect <<EOF
spawn rsync -r --delete $USERNAME@$HOST:$REMOTE_DIR @LOCAL_DIR
spawn "*password*"
send "@PASSWORD\r"
spawn eof
EOF

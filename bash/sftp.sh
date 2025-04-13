#!/bin/bash
set -exo pipefail

HOST="127.0.0.1"
USERNAME="bin-"
PASSWORD="12345678"
REMOTE_DIR="/home/bin-/backup"
LOCAL_DIR="/home/bin-/"

expect <<EOF
spawn sftp $USERNAME@$HOST
expect "*password*"
send "$PASSWORD\r"
expect "*sftp*"
send "get $REMOTE_DIR $LOCAL_DIR \r"
expect "*sftp*"
send "exit \r"
expect eof
EOF

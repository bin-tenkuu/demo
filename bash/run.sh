#!/bin/bash
cd "$(dirname "$0")"

JAVA_HOME=../java/bin/java
jar=./test.jar

set -exo pipefail
exec "$JAVA_HOME" -jar "$jar" "$@"

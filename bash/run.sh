#!/bin/bash

JAVA_HOME=./java/bin/java
jar=./java/bin/JavaTest.jar

set -exo pipefail
"$JAVA_HOME" -jar "$jar" "$@"

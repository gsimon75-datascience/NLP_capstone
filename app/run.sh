#!/usr/bin/env bash
THIS_PATH=$(readlink -f $(dirname -- $0))
#echo "library(shiny); options(shiny.trace = TRUE); runApp(appDir=\"$THIS_PATH\", host=\"127.0.0.1\", port=8080);" | R --no-save
echo "library(shiny); runApp(appDir=\"$THIS_PATH\", host=\"127.0.0.1\", port=8080);" | R --no-save


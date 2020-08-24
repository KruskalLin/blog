#!/bin/bash

git checkout gh-pages && rm -rf docs && gitbook build && cp -R _book/* docs/ && git add . && git commit -m "git commit $(date)" && git push && git checkout master

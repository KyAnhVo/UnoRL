#!/bin/bash

echo "Training:"
./train.py 2000000

echo "Push to github"
git add -A
git commit -m "trained and updated"
git push

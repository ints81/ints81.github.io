#!/bin/bash

commit_msg="'$*'"
date=$(date '+%Y-%m-%d %H:%M:%S')

cd $BLOG_HOME

git add .
git commit -m "${date} ${commit_msg}"
git push

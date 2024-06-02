#!/bin/bash

title=$1
date=$(date '+%Y-%m-%d')
cur_date_and_time=$(date '+%Y-%m-%d %H:%M:%S')

touch ${date}-${title}.md

cat <<EOF >${date}-${title}.md
---
title: TITLE
date: ${cur_date_and_time}
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase
toc: true
---
EOF

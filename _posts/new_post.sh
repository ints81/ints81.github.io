#!/bin/bash

date=$(date '+%Y-%m-%d')
cur_date_and_time=$(date '+%Y-%m-%d %H:%M:%S')

touch ${date}-TITLE.md

cat <<EOF >${date}-TITLE.md
---
title: TITLE
date: ${cur_date_and_time}
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase
toc: true
---
EOF

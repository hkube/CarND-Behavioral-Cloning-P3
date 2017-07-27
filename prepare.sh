#!/bin/bash

source activate carnd-term1
git config --global --add alias.tree "log --decorate --oneline"
git config --global --add alias.st status
git config --global --add alias.ci commit
git config --global --add alias.co checkout
git config --global --add user.email "harald.kube@gmx.de"
git config --global --add user.name  "Harald Kube"

wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

wget http://www.hkube.net/udacity/data1.tar.bz2
tar -xf data1.tar.bz2

#!/bin/bash

source activate carnd-term1
git config --global --add alias.tree "log --decorate --oneline"
git config --global --add alias.st status
git config --global --add alias.ci commit
git config --global --add alias.co checkout
git config --global --add user.email "harald.kube@gmx.de"
git config --global --add user.name  "Harald Kube"

for a in data2.tar.bz2; do
	wget http://www.hkube.net/udacity/$a
	tar -xf $a

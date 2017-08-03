#!/bin/bash


source activate carnd-term1
git config --global --add alias.tree "log --decorate --oneline"
git config --global --add alias.st status
git config --global --add alias.ci commit
git config --global --add alias.co checkout
git config --global --add user.email "harald.kube@gmx.de"
git config --global --add user.name  "Harald Kube"

#for a in data2 data3; do
#	echo Downloading $a
#	wget http://www.hkube.net/udacity/${a}.tar.bz2
#	tar -xf $a
#done

wget https://www.dropbox.com/sh/s0mfjceut1kllh8/AADmkdmeL_GLZzZcJvTII9Tka/data1.tar.bz2?dl=0 -O data1.tar.bz2
wget https://www.dropbox.com/sh/s0mfjceut1kllh8/AAD5DU1vYjX5ZWy2vKfSTHqUa/data2.tar.bz2?dl=0 -O data2.tar.bz2
wget https://www.dropbox.com/sh/s0mfjceut1kllh8/AADkZqrR4xwHukkYhvAFCIKba/data3.tar.bz2?dl=0 -O data3.tar.bz2

for a in *.tar.bz2; do
	tar -xf $a
done

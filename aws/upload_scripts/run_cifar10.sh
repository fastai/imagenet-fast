#!/bin/bash
cd ~/fastai
git fetch origin
git checkout origin/master

cd ~/cifar10
ln -rs ~/fastai/fastai ~/cifar10/fastai
echo "Linked fastai library"
source activate fastai
echo "Activated fastai conda"
python ~/cifar10/cifar10.py
echo "Ran cifar10 script"

# sudo halt
# sudo shutdown -h now
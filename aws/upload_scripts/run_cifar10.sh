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

# on demand instance
# shutdown -h now

# not one demand
sudo apt install ec2-api-tools
# ec2-terminate-instances $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
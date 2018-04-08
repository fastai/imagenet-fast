#!/bin/bash
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -p|--project_name)
    PROJECT="$2"
    shift # past argument
    shift # past value
    ;;
esac
done


if [[ -z ${PROJECT+x} ]]; then
    PROJECT="imagenet_training"
fi
TIME="(date +%s)"
PROJECT=$PROJECT-"$(date +%s)"

cd ~/fastai
git pull
git checkout fp16

conda activate fastai
conda env update
ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages

DATA_DIR=~/data/imagenet
DATA_DIR_160=~/data/imagenet

# Cleanup. Might not be a problem in newest AMI
sudo apt update && sudo apt install -y libsm6 libxext6
pip install torchtext
# Rogue files in validation set
rm ~/data/imagenet/val/make-data.py
rm ~/data/imagenet/val/valprep.sh
rm ~/data/imagenet/val/meta.pkl


cd ~/git/imagenet-fast/imagenet_nv
git pull

# Run single gpu
# python fastai.py ~/data/imagenet --arch resnext_50_32x4d -j 8 --epochs 1 -b 64 --fp16
# multi process
python multiproc.py -m fastai.py $DATA_DIR --arch resnext_50_32x4d -j 8 --epochs 1 -b 64 --world-size 4 --fp16

mkdir $PROJECT
cp -r $DATA_DIR/models $PROJECT
cp -r $DATA_DIR/training_logs $PROJECT
mkdir $PROJECT/imagenet-160
cp $DATA_DIR_160/models $PROJECT/imagenet-160
cp $DATA_DIR_160/training_logs $PROJECT/imagenet-160

scp -o StrictHostKeyChecking=no -r $PROJECT ubuntu@aws-m5.mine.nu:~/data/imagenet_training


echo Done. Powering off now
sudo shutdown --poweroff now
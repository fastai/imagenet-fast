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
    -sargs|--script_args)
    SARGS="$2"
    shift # past argument
    shift # past value
    ;;
    -multi|--use_multiproc)
    MULTI="$1"
    shift # past argument
    ;;
    -sh|--auto_shut)
    SHUTDOWN="$1"
    shift # past argument
    ;;
esac
done


if [[ -z ${PROJECT+x} ]]; then
    PROJECT="imagenet_training"
fi
if [[ -z ${SARGS+x} ]]; then
    echo "Must provide -sargs. E.G. '-a resnet50 -j 7 --epochs 100 -b 128 --loss-scale 128 --fp16 --world-size 8'"
    exit
fi
if [[ -n "$MULTI" ]]; then
    MULTI="-m multiproc"
fi
TIME="(date '+%Y-%m-%d-%H-%M-%S')"
PROJECT=$PROJECT-$TIME

# Warm up imagenet files?
tmux new-window -t imagenet -n 2 -d
tmux send-keys -t imagenet:2 "ls ~/data/imagenet-160/train" Enter
tmux send-keys -t imagenet:2 "ls ~/data/imagenet-160/val" Enter
tmux send-keys -t imagenet:2 "ls ~/data/imagenet/train" Enter
tmux send-keys -t imagenet:2 "ls ~/data/imagenet/val" Enter

cd ~/fastai
git pull
conda activate fastai
conda env update
ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages

DATA_DIR=~/data/imagenet
DATA_DIR_160=~/data/imagenet
SAVE_DIR=~/$PROJECT
mkdir $SAVE_DIR

# Cleanup. Might not be a problem in newest AMI
sudo apt update && sudo apt install -y libsm6 libxext6
pip install torchtext
# Rogue files in validation set
rm ~/data/imagenet/val/make-data.py
rm ~/data/imagenet/val/valprep.sh
rm ~/data/imagenet/val/meta.pkl


cd ~/git/imagenet-fast/imagenet_nv
git pull

# Run main.py
time python $MULTI main.py $DATA_DIR $SARGS |& tee -a output.log

mv *.log $SAVE_DIR
mv *.tar $SAVE_DIR
scp -o StrictHostKeyChecking=no -r $SAVE_DIR ubuntu@aws-m5.mine.nu:~/data/imagenet_training

if [[ -n "$SHUTDOWN" ]]; then
    echo Done. Shutting instance down
    sudo shutdown --poweroff now
else
    echo Done. Please remember to shut instance down when no longer needed.
fi
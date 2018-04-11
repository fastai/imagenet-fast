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
    -dir|--data_dir)
    DATA_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    -multi|--use_multiproc)
    MULTI="$1"
    shift # past argument
    ;;
    -warmup|--warmup_ebs)
    WARMUP="$1"
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
if [[ -z ${DATA_DIR+x} ]]; then
    DATA_DIR=~/data/imagenet
fi
if [[ -z ${SARGS+x} ]]; then
    echo "Must provide -sargs. E.G. '-a resnet50 -j 7 --epochs 100 -b 128 --loss-scale 128 --fp16 --world-size 8'"
    exit
fi
if [[ -n "$MULTI" ]]; then
    MULTI="-m multiproc"
fi
TIME=$(date '+%Y-%m-%d-%H-%M-%S')
PROJECT=$TIME-$PROJECT

cd ~/fastai
git pull
SHELL=/bin/bash
source ~/anaconda3/bin/activate fastai && conda env update -f=environment.yml
ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages

SAVE_DIR=~/$PROJECT
mkdir $SAVE_DIR

if [[ -n "$WARMUP" ]]; then
    echo "Warming up volume: $(date '+%Y-%m-%d-%H-%M-%S')" |& tee -a $SAVE_DIR/script.log
    sudo apt install fio -y
    if [[ $SARGS = *"train-128"* ]]; then
        sudo fio --directory=$DATA_DIR-160 --rw=randread --bs=128k --iodepth=32 --ioengine=libaio --direct=1 --name=volume-warmup -size=40G
        tmux new-window -t imagenet -n 2 -d
        tmux send-keys -t imagenet:2 "sudo fio --directory=$DATA_DIR --rw=randread --bs=128k --iodepth=32 --ioengine=libaio --direct=1 --name=volume-warmup -size=80G" Enter
    else
        sudo fio --directory=$DATA_DIR --rw=randread --bs=128k --iodepth=32 --ioengine=libaio --direct=1 --name=volume-warmup -size=40G
    fi
fi

OLD_BACKUP=$SAVE_DIR/backup
mkdir $OLD_BACKUP
mv *.log $OLD_BACKUP
mv *.tar $OLD_BACKUP

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
echo "Running script: time python $MULTI main.py $DATA_DIR --save-dir $SAVE_DIR $SARGS |& tee -a output.log" |& tee -a $SAVE_DIR/script.log
echo "Starting script: $(date '+%Y-%m-%d-%H-%M-%S')" |& tee -a $SAVE_DIR/script.log
time python $MULTI main.py $DATA_DIR --save-dir $SAVE_DIR $SARGS
echo "Script finished: $(date '+%Y-%m-%d-%H-%M-%S')" |& tee -a $SAVE_DIR/script.log

mv *.log $SAVE_DIR
mv *.tar $SAVE_DIR
scp -o StrictHostKeyChecking=no -r $SAVE_DIR ubuntu@aws-m5.mine.nu:~/data/imagenet_training

if [[ -n "$SHUTDOWN" ]]; then
    echo Done. Shutting instance down
    sudo shutdown --poweroff now
else
    echo Done. Please remember to shut instance down when no longer needed.
fi
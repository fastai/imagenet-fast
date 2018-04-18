#!/bin/bash
echo 'Starting script'

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -sargs|--script_args)
    SARGS="$2"
    shift
    ;;
    -p|--project_name)
    PROJECT="$2"
    shift
    ;;
    -sh|--auto_shut)
    SHUTDOWN="$1"
    ;;
esac
shift
done

if [[ -z ${SARGS+x} ]]; then
    echo "Must provide -sargs. E.G. '-a resnet50 -j 7 --epochs 100'"
    exit
fi
if [[ -z ${PROJECT+x} ]]; then
    PROJECT="imagenet_training"
fi
DATA_DIR=~/data/imagenet

TIME=$(date '+%Y-%m-%d-%H-%M-%S')
PROJECT=$TIME-$PROJECT
SAVE_DIR=~/$PROJECT
mkdir $SAVE_DIR

echo "$(date '+%Y-%m-%d-%H-%M-%S') Instance loaded. Updating projects." |& tee -a $SAVE_DIR/script.log
cd ~/fastai
git stash
git pull
git stash pop
SHELL=/bin/bash
source ~/anaconda3/bin/activate fastai && conda env update -f=environment.yml
ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages
cd ~/git/imagenet-fast/imagenet_nv
git pull
git checkout custom_script

cd ~/data/imagenet
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh
cd ../imagenet-sz/160/
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh
cd ../320/
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh

cd ~/data/imagenet/val
cp -r --parents */*1.JPEG ../val2/
cd ~/data/imagenet-sz/160/val
mkdir ../val2
cp -r --parents */*1.JPEG ../val2/
cd ~/data/imagenet-sz/320/val
mkdir ../val2
cp -r --parents */*1.JPEG ../val2/

cd ~/git/imagenet-fast/imagenet_nv
echo "$(date '+%Y-%m-%d-%H-%M-%S') Warming up volume." |& tee -a $SAVE_DIR/script.log
python -m multiproc jh_warm.py ~/data/imagenet -j 8 -a fa_resnet50 --fp16

echo "$(date '+%Y-%m-%d-%H-%M-%S') Running script: $SAVE_DIR $SARGS" |& tee -a $SAVE_DIR/script.log
time python -m multiproc main.py $DATA_DIR -j 8 --fp16 --epochs 1 -b 192 --loss-scale 512 --save-dir $SAVE_DIR $SARGS |& tee -a $SAVE_DIR/output.log
echo "$(date '+%Y-%m-%d-%H-%M-%S') Training finished." |& tee -a $SAVE_DIR/script.log
scp -o StrictHostKeyChecking=no -r $SAVE_DIR ubuntu@aws-m5.mine.nu:~/data/imagenet_training

if [[ -n "$SHUTDOWN" ]]; then
    echo Done. Shutting instance down
    sudo halt
else
    echo Done. Please remember to shut instance down when no longer needed.
fi


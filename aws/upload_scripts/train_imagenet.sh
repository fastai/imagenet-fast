#!/bin/bash
echo 'Starting script'

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
# Date before project name. That way logs show up in chronological order
TIME=$(date '+%Y-%m-%d-%H-%M-%S')
PROJECT=$TIME-$PROJECT
SAVE_DIR=~/$PROJECT
mkdir $SAVE_DIR

echo "$(date '+%Y-%m-%d-%H-%M-%S') Instance loaded. Updating projects." |& tee -a $SAVE_DIR/script.log
cd ~/fastai
git stash
git pull
git checkout fp16
git stash pop
SHELL=/bin/bash
source ~/anaconda3/bin/activate fastai && conda env update -f=environment.yml
ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages
cd ~/git/imagenet-fast/imagenet_nv
git pull
git checkout as_test

# Rogue files in validation set
rm ~/data/imagenet/val/make-data.py
rm ~/data/imagenet/val/valprep.sh
rm ~/data/imagenet/val/meta.pkl
# This image is originally a PNG. However, it did not get converted right in the resize. Ignore the iamge until AMI dataset is fixed
mkdir $DATA_DIR-160/broken
mv $DATA_DIR-160/train/n02105855/n02105855_2933.JPEG $DATA_DIR-160/broken

if [[ -n "$WARMUP" ]]; then
    echo "$(date '+%Y-%m-%d-%H-%M-%S') Warming up volume." |& tee -a $SAVE_DIR/script.log
    python -m multiproc jh_warm.py ~/data/imagenet -j 8 -a fa_resnet50 -b 256 --fp16
fi

cd ~/data/imagenet
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh
cd ../imagenet-sz/160/
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh
cd ../320/
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh
cd ~/git/imagenet-fast/imagenet_nv

# Run fastai_imagenet
echo "$(date '+%Y-%m-%d-%H-%M-%S') Running script: time python $MULTI as_tmp.py $DATA_DIR --save-dir $SAVE_DIR $SARGS" |& tee -a $SAVE_DIR/script.log
time python $MULTI as_tmp.py $DATA_DIR --save-dir $SAVE_DIR $SARGS |& tee -a $SAVE_DIR/output.log
echo "$(date '+%Y-%m-%d-%H-%M-%S') Imagenet training finished." |& tee -a $SAVE_DIR/script.log

scp -o StrictHostKeyChecking=no -r $SAVE_DIR ubuntu@aws-m5.mine.nu:~/data/imagenet_training

if [[ -n "$SHUTDOWN" ]]; then
    echo Done. Shutting instance down
    sudo shutdown --poweroff now
else
    echo Done. Please remember to shut instance down when no longer needed.
fi


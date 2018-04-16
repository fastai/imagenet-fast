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
    -resume|--resume_h5_file)
    RESUME="$2"
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
git stash pop
SHELL=/bin/bash
source ~/anaconda3/bin/activate fastai && conda env update -f=environment.yml
ln -s ~/fastai/fastai ~/anaconda3/envs/fastai/lib/python3.6/site-packages
cd ~/git/imagenet-fast/imagenet_nv
git pull

# Cleanup. Might not be a problem in newest AMI
sudo apt update && sudo apt install -y libsm6 libxext6
pip install torchtext
pip uninstall pillow --yes
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
# Rogue files in validation set
rm ~/data/imagenet/val/make-data.py
rm ~/data/imagenet/val/valprep.sh
rm ~/data/imagenet/val/meta.pkl
# This image is originally a PNG. However, it did not get converted right in the resize. Ignore the iamge until AMI dataset is fixed
mkdir $DATA_DIR-160/broken
mv $DATA_DIR-160/train/n02105855/n02105855_2933.JPEG $DATA_DIR-160/broken

if [[ -n "$WARMUP" ]]; then
    echo "$(date '+%Y-%m-%d-%H-%M-%S') Warming up volume." |& tee -a $SAVE_DIR/script.log
    # libaio engine enables async random reads/writes. 
    # If server bootup time doesn't matter, we can do a complete warmup by using dd and copy $DATA_DIR to /dev/null
    # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-initialize.html 
    # sudo dd if=$DATA_DIR of=/dev/null
    sudo apt install fio -y
    if [[ $SARGS = *"train-128"* ]]; then
        sudo fio --directory=$DATA_DIR-160 --rw=randread --bs=128k --iodepth=32 --ioengine=libaio --direct=1 --name=volume-warmup -size=40G
        tmux new-window -t imagenet -n 2 -d
        tmux send-keys -t imagenet:2 "sudo fio --directory=$DATA_DIR --rw=randread --bs=128k --iodepth=32 --ioengine=libaio --direct=1 --name=volume-warmup -size=80G" Enter
    else
        sudo fio --directory=$DATA_DIR --rw=randread --bs=128k --iodepth=32 --ioengine=libaio --direct=1 --name=volume-warmup -size=40G
    fi
fi


if [[ -n "$RESUME" ]]; then
    echo Resuming from file $RESUME
    RESUME_DIR=$SAVE_DIR/resume
    mkdir $RESUME_DIR
    scp -o StrictHostKeyChecking=no ubuntu@aws-m5.mine.nu:$RESUME $RESUME_DIR
    SARGS="$SARGS --resume $RESUME_DIR/$(basename $RESUME)"
fi
# Run fastai_imagenet
echo "$(date '+%Y-%m-%d-%H-%M-%S') Running script: time python $MULTI fastai_imagenet.py $DATA_DIR --save-dir $SAVE_DIR $SARGS" |& tee -a $SAVE_DIR/script.log
time python $MULTI fastai_imagenet.py $DATA_DIR --save-dir $SAVE_DIR $SARGS |& tee -a $SAVE_DIR/output.log
echo "$(date '+%Y-%m-%d-%H-%M-%S') Imagenet training finished." |& tee -a $SAVE_DIR/script.log

scp -o StrictHostKeyChecking=no -r $SAVE_DIR ubuntu@aws-m5.mine.nu:~/data/imagenet_training

if [[ -n "$SHUTDOWN" ]]; then
    echo Done. Shutting instance down
    sudo shutdown --poweroff now
else
    echo Done. Please remember to shut instance down when no longer needed.
fi
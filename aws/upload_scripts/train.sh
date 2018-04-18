#!/bin/bash
echo 'Starting script'

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -s|--script)
    SCRIPT="$2"
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

# Rogue files in validation set
rm ~/data/imagenet/val/make-data.py
rm ~/data/imagenet/val/valprep.sh
rm ~/data/imagenet/val/meta.pkl
# This image is originally a PNG. However, it did not get converted right in the resize. Ignore the iamge until AMI dataset is fixed
mkdir $DATA_DIR-160/broken
mv $DATA_DIR-160/train/n02105855/n02105855_2933.JPEG $DATA_DIR-160/broken

if [[ -n "$WARMUP" ]]; then
    echo "$(date '+%Y-%m-%d-%H-%M-%S') Warming up volume." |& tee -a $SAVE_DIR/script.log
    python -m multiproc jh_warm.py ~/data/imagenet -j 8 -a fa_resnet50 --fp16
fi

cd ~/data/imagenet
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh
cd ../imagenet-sz/160/
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh
cd ../320/
bash ~/git/imagenet-fast/imagenet_nv/blacklist.sh
cd ~/git/imagenet-fast/imagenet_nv

if [[ ! -z $SCRIPT ]]; then
    echo "$(date '+%Y-%m-%d-%H-%M-%S') Running script: $SCRIPT $SAVE_DIR" |& tee -a $SAVE_DIR/script.log
    bash $SCRIPT $SAVE_DIR |& tee -a $SAVE_DIR/output.log
    echo "$(date '+%Y-%m-%d-%H-%M-%S') Training finished." |& tee -a $SAVE_DIR/script.log
    scp -o StrictHostKeyChecking=no -r $SAVE_DIR ubuntu@aws-m5.mine.nu:~/data/imagenet_training
fi

if [[ -n "$SHUTDOWN" ]]; then
    echo Done. Shutting instance down
    sudo halt
else
    echo Done. Please remember to shut instance down when no longer needed.
fi


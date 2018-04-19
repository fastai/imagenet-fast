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
    -dir|--data_dir)
    DATA_DIR="$2"
    shift
    ;;
    -b)
    BS="$2"
    shift
    ;;
    --sz)
    SIZE="$2"
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
if [[ -z ${BS+x} ]]; then
    BS=192
fi
if [[ -z ${SIZE+x} ]]; then
    SIZE=224
fi
if [[ -z ${DATA_DIR+x} ]]; then
    DATA_DIR=~/data/imagenet
fi

TIME=$(date '+%Y-%m-%d-%H-%M-%S')
PROJECT=$TIME-$PROJECT
SAVE_DIR=~/$PROJECT
mkdir $SAVE_DIR

cd ~/git/imagenet-fast/imagenet_nv

echo "$(date '+%Y-%m-%d-%H-%M-%S') Running script: $SAVE_DIR $SARGS" |& tee -a $SAVE_DIR/script.log
echo python -m multiproc main.py $DATA_DIR --sz $SIZE -j 8 --fp16 --epochs 80 -b $BS --loss-scale 512 --save-dir $SAVE_DIR $SARGS
time python -m multiproc main.py $DATA_DIR --sz $SIZE -j 8 --fp16 --epochs 80 -b $BS --loss-scale 512 --save-dir $SAVE_DIR $SARGS |& tee -a $SAVE_DIR/output.log
echo "$(date '+%Y-%m-%d-%H-%M-%S') Training finished." |& tee -a $SAVE_DIR/script.log


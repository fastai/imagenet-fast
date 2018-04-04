#!/bin/bash
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -d|--device)
    DEVICE="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--reformat)
    REFORMAT="$2"
    shift # past argument
    shift # past value
    ;;
    -dir|--directory)
    DIRECTORY="$2"
    shift # past argument
    shift # past value
    ;;
esac
done


if [[ -z ${DIRECTORY+x} ]]; then
    DIRECTORY="ebs_mount_point"
fi

set -- "${POSITIONAL[@]}" # restore positional parameters

if [[ -z ${DEVICE+x} ]]; then
    echo "Must provide --device. E.G. /dev/xvdf"
    exit
fi

if [[ -n "$REFORMAT" ]]; then
    yes | sudo mkfs -t ext4 $DEVICE
fi

if [ ! -d $DIRECTORY ]; then
        sudo mkdir $DIRECTORY
fi

sudo mount $DEVICE $DIRECTORY
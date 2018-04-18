### Script

In the parent directory (../cifar), run:

    time python -m multiproc.py dawn_mod.py /home/ubuntu/data/cifar10 \
    --save-dir /home/ubuntu/data/cf_train_save/wrn_submission_v6 \
    -a wrn_22 \
    --epochs 1 --cycle-len 50 \
    --fp16 --loss-scale 512 \
    -b 128 \
    --wd 2e-4 --lr 1.3 \
    --use-clr 50,15,0.95,0.85

on an Amazon p3.16xlarge instance.  You will need to symlink a fastai install inside the cifar directory.

Dataset from:

    http://files.fast.ai/data/cifar10.tgz




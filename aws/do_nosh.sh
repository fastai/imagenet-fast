python run_script.py -zone us-west-2a --launch-method demand -iname rn50_30_45 --run-script upload_scripts/train2.sh -sargs "-sargs '-a resnet50 --lr 0.30 --epochs 45'" & sleep 5
python run_script.py -zone us-west-2a --launch-method demand -iname rn50_40_45 --run-script upload_scripts/train2.sh -sargs "-sargs '-a resnet50 --lr 0.40 --epochs 45'" & sleep 5


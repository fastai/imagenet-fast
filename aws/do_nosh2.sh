
python run_script.py -zone us-west-2a --launch-method demand -iname w125_20_65 --run-script upload_scripts/train2.sh -sargs "-sargs '-a w125_resnet50 --lr 0.20 --epochs 65'" & sleep 5
python run_script.py -zone us-west-2b --launch-method demand -iname w125_60_45 --run-script upload_scripts/train2.sh -sargs "-sargs '-a w125_resnet50 --lr 0.60 --epochs 45'" & sleep 5
python run_script.py -zone us-west-2b --launch-method demand -iname w125_60_55 --run-script upload_scripts/train2.sh -sargs "-sargs '-a w125_resnet50 --lr 0.60 --epochs 55'" & sleep 5
python run_script.py -zone us-west-2b --launch-method demand -iname w125_60_65 --run-script upload_scripts/train2.sh -sargs "-sargs '-a w125_resnet50 --lr 0.60 --epochs 65'" & sleep 5


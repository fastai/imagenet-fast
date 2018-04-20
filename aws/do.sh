python run_script.py --launch-method demand -iname rn50_30_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.30 --epochs 45'" & sleep 5
python run_script.py --launch-method demand -iname rn50_30_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.30 --epochs 55'" & sleep 5
python run_script.py --launch-method demand -iname rn50_30_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.30 --epochs 65'" & sleep 5

python run_script.py --launch-method demand -iname rn50_40_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.40 --epochs 45'" & sleep 5
python run_script.py --launch-method demand -iname rn50_40_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.40 --epochs 55'" & sleep 5
python run_script.py --launch-method demand -iname rn50_40_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.40 --epochs 65'"

python run_script.py --launch-method demand -iname w125_30_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.30 --epochs 45'" & sleep 5
python run_script.py --launch-method demand -iname w125_30_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.30 --epochs 55'" & sleep 5
python run_script.py --launch-method demand -iname w125_30_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.30 --epochs 65'" & sleep 5

python run_script.py --launch-method demand -iname w125_40_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.40 --epochs 45'" & sleep 5
python run_script.py --launch-method demand -iname w125_40_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.40 --epochs 55'" & sleep 5
python run_script.py --launch-method demand -iname w125_40_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.40 --epochs 65'"

python run_script.py -zone us-west-2c --launch-method demand -iname rn50_20_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.20 --epochs 45'" & sleep 5
python run_script.py -zone us-west-2c --launch-method demand -iname rn50_20_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.20 --epochs 55'" & sleep 5
python run_script.py -zone us-west-2c --launch-method demand -iname rn50_20_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.20 --epochs 65'" & sleep 5

python run_script.py -zone us-west-2c --launch-method demand -iname rn50_60_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.60 --epochs 45'" & sleep 5
python run_script.py -zone us-west-2c --launch-method demand -iname rn50_60_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.60 --epochs 55'" & sleep 5
python run_script.py -zone us-west-2c --launch-method demand -iname rn50_60_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.60 --epochs 65'"

python run_script.py -zone us-west-2c --launch-method demand -iname w125_20_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.20 --epochs 45'" & sleep 5
python run_script.py -zone us-west-2c --launch-method demand -iname w125_20_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.20 --epochs 55'" & sleep 5
python run_script.py -zone us-west-2c --launch-method demand -iname w125_20_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.20 --epochs 65'" & sleep 5

python run_script.py -zone us-west-2c --launch-method demand -iname w125_60_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.60 --epochs 45'" & sleep 5
python run_script.py -zone us-west-2c --launch-method demand -iname w125_60_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.60 --epochs 55'" & sleep 5
python run_script.py -zone us-west-2c --launch-method demand -iname w125_60_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.60 --epochs 65'"


python run_script.py --launch-method demand -iname rn50_20_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.20 --epochs 45'"
python run_script.py --launch-method demand -iname rn50_20_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.20 --epochs 55'"
python run_script.py --launch-method demand -iname rn50_20_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.20 --epochs 65'"

python run_script.py --launch-method demand -iname rn50_60_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.60 --epochs 45'"
python run_script.py --launch-method demand -iname rn50_60_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.60 --epochs 55'"
python run_script.py --launch-method demand -iname rn50_60_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.60 --epochs 65'"

python run_script.py --launch-method demand -iname w125_20_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.20 --epochs 45'"
python run_script.py --launch-method demand -iname w125_20_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.20 --epochs 55'"
python run_script.py --launch-method demand -iname w125_20_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.20 --epochs 65'"

python run_script.py --launch-method demand -iname w125_60_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.60 --epochs 45'"
python run_script.py --launch-method demand -iname w125_60_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.60 --epochs 55'"
python run_script.py --launch-method demand -iname w125_60_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.60 --epochs 65'"


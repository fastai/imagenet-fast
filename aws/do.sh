python run_script.py --launch-method demand -iname rn50_30_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.30 --epochs 45'"
python run_script.py --launch-method demand -iname rn50_30_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.30 --epochs 55'"
python run_script.py --launch-method demand -iname rn50_30_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.30 --epochs 65'"

python run_script.py --launch-method demand -iname rn50_40_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.40 --epochs 45'"
python run_script.py --launch-method demand -iname rn50_40_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.40 --epochs 55'"
python run_script.py --launch-method demand -iname rn50_40_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.40 --epochs 65'"

python run_script.py --launch-method demand -iname w125_30_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.30 --epochs 45'"
python run_script.py --launch-method demand -iname w125_30_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.30 --epochs 55'"
python run_script.py --launch-method demand -iname w125_30_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.30 --epochs 65'"

python run_script.py --launch-method demand -iname w125_40_45 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.40 --epochs 45'"
python run_script.py --launch-method demand -iname w125_40_55 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.40 --epochs 55'"
python run_script.py --launch-method demand -iname w125_40_65 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.40 --epochs 65'"


python run_script.py --launch-method demand -iname rn50_10 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.1'"
python run_script.py --launch-method demand -iname rn50_13 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.13'"
python run_script.py --launch-method demand -iname rn50_20 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.20'"
python run_script.py --launch-method demand -iname rn50_30 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.30'"
python run_script.py --launch-method demand -iname rn50_40 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a resnet50 --lr 0.40'"

python run_script.py --launch-method demand -iname bnz_20 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a bnz_resnet50 --lr 0.20'"
python run_script.py --launch-method demand -iname bnz_30 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a bnz_resnet50 --lr 0.30'"
python run_script.py --launch-method demand -iname bnz_40 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a bnz_resnet50 --lr 0.40'"

python run_script.py --launch-method demand -iname w15_13 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w15_resnet50 --lr 0.13'"
python run_script.py --launch-method demand -iname w125_13 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a w125_resnet50 --lr 0.13'"

python run_script.py --launch-method demand -iname rn50_13_wd35 --run-script upload_scripts/train2.sh -sargs \
	"-sh -sargs '--wd 3e-5 -a resnet50 --lr 0.13'"


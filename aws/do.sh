#python run_script.py --launch-method demand -iname bnf_10 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a bnfinal_resnet50 --lr 0.1'"

#python run_script.py --launch-method demand -iname fa4_10 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a fa4_resnet50 --lr 0.1'"
#python run_script.py --launch-method demand -iname fa4_13 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a fa4_resnet50 --lr 0.13'"
python run_script.py --launch-method demand -iname fa4_30 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a fa4_resnet50 --lr 0.30'"
python run_script.py --launch-method demand -iname fa4_40 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a fa4_resnet50 --lr 0.40'"

#python run_script.py --launch-method demand -iname fa5_20 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a fa5_resnet50 --lr 0.2'"
#python run_script.py --launch-method demand -iname fa5_07 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a fa5_resnet50 --lr 0.07'"
#python run_script.py --launch-method demand -iname fa5_13 --run-script upload_scripts/train2.sh -sargs "-sh -sargs '-a fa5_resnet50 --lr 0.13'"


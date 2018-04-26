python run_script.py -zone us-west-2b --launch-method find -iname p362 -p rn50_25_45_bnf_main_sml --run-script upload_scripts/train2.sh -sargs "-sargs '-a resnet50 --lr 0.25 --epochs 45 --small'" &
python run_script.py -zone us-west-2b --launch-method find -iname p36 -p rn50_40_45_bnf_main_sml --run-script upload_scripts/train2.sh -sargs "-sargs '-a resnet50 --lr 0.40 --epochs 45 --small'" &
#sleep 10
#python run_script.py -zone us-west-2b --launch-method find -iname vggrn50_30_45_bnf_main_sml --run-script upload_scripts/train2.sh -sargs "-sargs '-a vgg_resnet50 --lr 0.30 --epochs 45 --small'" &
#sleep 10
#python run_script.py -zone us-west-2b --launch-method find -iname rn50_30_40_bnf_main_sml --run-script upload_scripts/train2.sh -sargs "-sargs '-a resnet50 --lr 0.30 --epochs 40 --small'" &
#sleep 10
#python run_script.py -zone us-west-2b --launch-method find -iname vggrn50_30_40_bnf_main_sml --run-script upload_scripts/train2.sh -sargs "-sargs '-a vgg_resnet50 --lr 0.30 --epochs 40 --small'" &
#sleep 10
#python run_script.py -zone us-west-2b --launch-method find -iname rn50_30_40_bnf_main_sml_wd5 --run-script upload_scripts/train2.sh -sargs "-sargs '-a resnet50 --lr 0.30 --epochs 40 --small --wd 5e-5'" &
#sleep 10
#python run_script.py -zone us-west-2b --launch-method find -iname vggrn50_30_40_bnf_main_sml_wd5 --run-script upload_scripts/train2.sh -sargs "-sargs '-a vgg_resnet50 --lr 0.30 --epochs 40 --small --wd 5e-5'" &
#sleep 10


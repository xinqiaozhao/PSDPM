
# Step 1. Train psdpm for localization maps.
# 1.1 train psdpm
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_resnet50_psdpm.py --session_name exp
##  1.2 obtain localization maps
CUDA_VISIBLE_DEVICES=0,1,2,3 python make_cam.py --session_name exp
### # 1.3 evaluate localization maps
CUDA_VISIBLE_DEVICES=2 python eval_cam.py --session_name exp
#

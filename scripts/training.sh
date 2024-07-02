GPUID=0

cd ..

# DVERGE training
CUDA_VISIBLE_DEVICES=2 python train/train_combine.py --model-num 3 --distill-eps 0.07 --distill-alpha 0.007 --start-from 'baseline' --distill-steps 5 --fb 0.8
# python train/train_cross.py --gpu 1  --model-num 3 --eps 0.09 --distill-eps 0.09 # PGDLinf
# python train/train_fixed-distill.py  --model-num 3 --eps 0.09 --distill-eps 0.09 # fixed

CUDA_VISIBLE_DEVICES=2 python train/train_dverge.py --model-num 3 --distill-eps 0.05 --distill-alpha 0.05 --start-from 'baseline' --distill-steps 5
# Baseline training
CUDA_VISIBLE_DEVICES=4 python train/train_baseline.py --model-num 1
nohup python train/train_baseline.py >> logs/naive_advtrain.log 2>&1 &

# ADP training
CUDA_VISIBLE_DEVICES=6 python train/train_adp.py  --model-num 3

# GAL training
CUDA_VISIBLE_DEVICES=5 python train/train_gal.py  --model-num 3

# GAL training
CUDA_VISIBLE_DEVICES=5 python train/train_adv.py  --model-num 3 --fb 0.6

### FDT
CUDA_VISIBLE_DEVICES=0 python train/train_combine.py --model-num 3 --distill-eps 0.05 --distill-alpha 0.005 --start-from 'baseline' --distill-steps 5 --fb 0.4 --dateset cifar100 --arch wrn
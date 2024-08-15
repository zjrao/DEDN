#CUB GZSL
python Cub_train.py --batch_size 50 --epochs 80 --seed 214 --is_balance --lamb 0.0 \
--dim_f 2048 --dim_v 300 --dim_r 14 --trainable_w2v --is_bias --mal --bias 1 --normalize_F --lr 0.0001 \
--weight_decay 0.0001 --momentum 0.9 --p_ce 0.1 --p_crc 0.001 --c_mix 0.8 --e_mix 0.9

#CUB ZSL
#python Cub_train.py --batch_size 50 --epochs 80 --seed 214 --is_balance --lamb 0.18 \
#--dim_f 2048 --dim_v 300 --dim_r 14 --trainable_w2v --is_bias --mal --bias 1 --normalize_F --lr 0.0001 \
#--weight_decay 0.0001 --momentum 0.9 --p_ce 0.1 --p_crc 0.001 --c_mix 0.6 --e_mix 0.9

#SUN  GZSL
#python Sun_train.py --batch_size 50 --epochs 80 --seed 2339 --lamb 0.0 \
#--dim_f 2048 --dim_v 300 --dim_r 14 --trainable_w2v --is_bias --mal --bias 1 --normalize_F --lr 0.0001 \
#--weight_decay 0.0001 --momentum 0.9 --p_ce 0.1 --p_crc 0.001 --c_mix 0.95 --e_mix 0.3 --hidd_f 256

#SUN ZSL
#python Sun_train.py --batch_size 50 --epochs 80 --seed 2339 --lamb 0.0 \
#--dim_f 2048 --dim_v 300 --dim_r 14 --trainable_w2v --is_bias --mal --bias 1 --normalize_F --lr 0.0001 \
#--weight_decay 0.0001 --momentum 0.9 --p_ce 0.1 --p_crc 0.001 --c_mix 0.95 --e_mix 0.3 --hidd_f 128

#AWA2 GZSL
#python Awa2_train.py --batch_size 50 --epochs 20 --seed 87778 --is_balance --lamb 0.11 \
#--dim_f 2048 --dim_v 300 --dim_r 7 --trainable_w2v --is_bias --mal --bias 1 --normalize_F --lr 0.0001 \
#--weight_decay 0.0001 --momentum 0.9 --p_ce 0.1 --p_crc 0.001 --c_mix 0.8 --e_mix 0.5

#AWA2 ZSL
#python Awa2_train.py --batch_size 50 --epochs 20 --seed 87778 --lamb 0.11 \
#--dim_f 2048 --dim_v 300 --dim_r 7 --trainable_w2v --is_bias --mal --bias 1 --normalize_F --lr 0.0001 \
#--weight_decay 0.0001 --momentum 0.9 --p_ce 0.1 --p_crc 0.001 --c_mix 0.8 --e_mix 0.5

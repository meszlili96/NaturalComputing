python3.7 ../train_opt.py \
--workers 1 \
--batch_size 64 \
--num_epochs 100 \
--lr 0.0001 \
--beta1 0.5 \
--beta2 0.999 \
--ngpu 0 \
--toy_type 1 \
--toy_std 0.02 \
--toy_scale 2.0 \
--toy_len 32000 \
--g_loss 3 \
--gan_type "EGAN" \
--gamma 0.05


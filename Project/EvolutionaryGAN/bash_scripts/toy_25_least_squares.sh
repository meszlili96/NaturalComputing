python3.7 ../train_opt.py \
--workers 1 \
--batch_size 64 \
--num_epochs 100 \
--lr 0.00005 \
--beta1 0.5 \
--beta2 0.999 \
--ngpu 0 \
--toy_type 2 \
--toy_std 0.05 \
--toy_scale 2.0 \
--toy_len 32000 \
--g_loss 3 \
--gan_type "GAN"


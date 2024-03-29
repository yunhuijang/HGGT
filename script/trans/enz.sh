python trainer/train_trans_generator.py \
--order C-M \
--dataset_name GDSS_enz \
--max_epochs 500 \
--check_sample_every_n_epoch 501 \
--replicate 1 \
--max_len 238 \
--wandb_on online \
--string_type group-red \
--lr 0.0002 \
--batch_size 64 \
--sample_batch_size 64 \
--num_samples 64 \
--dropout 0.1 \
--input_dropout 0 \
--k 2
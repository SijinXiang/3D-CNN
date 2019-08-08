cd ..
python run.py \
    --pretrained_model pretrain/model.ckpt-603 \
    --input_seq_length 6 \
    --output_seq_length 6 \
    --num_hidden 10,10,10,5 \
    --kernel_size 20 \
    --pool_size 1,5,5 \
    --strides 1 \
    --lr 0.0001 \
    --batch_size 12 \
    --max_iterations 5000 \
    --is_training True \
    --save_dir checkpoints \
    --gen_frm_dir results \
    --train_data_paths data/milan_tra.npy \
    --valid_data_paths data/milan_val.npy \
    --test_data_paths data/milan_test.npy \
    --dataset_name milan \
    --img_width 100 \
    --model_name 3dcnn \
    --display_interval 1 \
    --test_interval 1 \
    --snapshot_interval 1 \
    --allow_gpu_growth True
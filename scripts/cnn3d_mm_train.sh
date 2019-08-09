#    --pretrained_model checkpoints/model.ckpt-2 \
cd ..
python run.py \
    --input_seq_length 6 \
    --output_seq_length 6 \
    --num_hidden 10,10 \
    --kernel_size 20 \
    --pool_size 1,5,5 \
    --strides 1 \
    --lr 0.001 \
    --batch_size 40 \
    --max_iterations 10000 \
    --is_training True \
    --save_dir checkpoints/6-6/10-10 \
    --gen_frm_dir results/6-6/10-10 \
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

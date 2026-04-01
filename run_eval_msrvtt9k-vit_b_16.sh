CUDA_VISIBLE_DEVICES=0

time=$(date +"%Y_%m_%d-%H_%M_%S")
ANNO_PATH=$PWD/data/origin/MSRVTT/msrvtt_data/
VID_PATH=$PWD/data/origin/MSRVTT/videos/all/
BEST_CKPT_FOLDER=PATH_TO_FOLDER_OF_BEST_CKPT
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main_my.py \
    --do_train 0 --eval_in_train 0 --do_eval 1 --seed 0 --num_thread_reader 0 \
    --epochs 5 --batch_size 32 --n_display 50 \
    --output_dir $BEST_CKPT_FOLDER/eval-msrvtt9k-vit_b_16/ --datatype msrvtt9k --expand_msrvtt_sentences 1 \
    --train_csv ${ANNO_PATH}/MSRVTT_train.9k.csv \
    --val_csv ${ANNO_PATH}/MSRVTT_JSFUSION_test.csv \
    --data_path ${ANNO_PATH}/MSRVTT_data.json \
    --features_path ${VID_PATH} \
    --cross_num_hidden_layers 4 \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 8 \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type 1 --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 \
    --from_script 1 \
    --support_loss_weight 0.8 \
    --eam_support_loss_weight 1.0 \
    --loss_fn sig --framepe 1 --energy_fn bilinear --energy_pooling avg \
    --best_ckpt_path $BEST_CKPT_FOLDER/msrvtt9k-vit_b_16.bin

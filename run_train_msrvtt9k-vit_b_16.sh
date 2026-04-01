CUDA_VISIBLE_DEVICES=0,1

time=$(date +"%Y_%m_%d-%H_%M_%S")
ANNO_PATH=$PWD/data/MSRVTT/msrvtt_data/
VID_PATH=$PWD/data/MSRVTT/videos/all/
CLIP_MODEL=16
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m torch.distributed.launch --nproc_per_node=2 --master_port=25001 \
    main_my.py --do_train 1 --eval_in_train 1 --num_thread_reader=8 --seed 0 \
    --epochs 5 --batch_size 64 --n_display 50 \
    --output_dir $PWD/ckpts/msrvtt9k/ViT_B_$CLIP_MODEL-$time --datatype msrvtt9k --expand_msrvtt_sentences 1 \
    --train_csv ${ANNO_PATH}/MSRVTT_train.9k.csv \
    --val_csv ${ANNO_PATH}/MSRVTT_JSFUSION_test.csv \
    --data_path ${ANNO_PATH}/MSRVTT_data.json \
    --features_path ${VID_PATH} \
    --cross_num_hidden_layers 4 \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 8 \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type 1 --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/$CLIP_MODEL \
    --from_script 1 \
    --support_loss_weight 0.8 \
    --eam_support_loss_weight 1.0 \
    --gnn_type rgat \
    --loss_fn sig --framepe 1 --energy_fn bilinear --energy_pooling avg

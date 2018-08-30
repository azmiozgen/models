python train.py \
    --dataset="check_localization" \
    --logtostderr \
    --training_number_of_steps=300000 \
    --log_steps=50 \
    --save_interval_secs=1000 \
    --save_summaries_secs=500 \
    --save_summaries_images=True \
    --train_split="train" \
    --model_variant="xception_65" \
    --learning_policy="poly" \
    --base_learning_rate=.01 \
    --learning_rate_decay_factor=0.1 \
    --learning_rate_decay_step=3000 \
    --learning_power=0.9 \
    --momentum=0.9 \
    --weight_decay=.00004 \
    --initialize_last_layer=False \
    --last_layers_contain_logits_only=True \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=1000 \
    --train_crop_size=1000 \
    --train_batch_size=4 \
    --fine_tune_batch_norm=False \
    --tf_initial_checkpoint="models/check_localization_mobilenetv2_coco_voc/model.ckpt-25921.index" \
    --train_logdir="models/check_localization_mobilenetv2_coco_voc" \
    --dataset_dir="datasets/check_localization/tfrecord"

## --tf_initial_checkpoint="../slim/checkpoints/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt" \
## --tf_initial_checkpoint="datasets/pascal_voc_seg/inception_resnet_v2/model.ckpt-0.data-00000-of-00001" \



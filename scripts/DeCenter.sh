torchrun --nproc_per_node=2 --master_port=29502 main_aitod.py \
  --dataset_file visdrone --output_dir outputs/test1 -c config/DeCenter_5scale.py --coco_path data/VisDrone \
  --pretrain_model_path pretrain_models/pretrain_model.pth \
  --finetune_ignore label_enc.weight class_embed \
  --options dn_scalar=100 embed_init_tgt=False \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0 \

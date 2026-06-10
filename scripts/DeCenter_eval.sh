checkpoint=$1
python main_aitod.py \
  --output_dir logs/DQ_eval \
	--dataset_file visdrone -c config/DQ_5scale.py --coco_path data/VisDrone \
	--eval --save_results  --resume $checkpoint \
	--options dn_scalar=100 embed_init_tgt=False \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

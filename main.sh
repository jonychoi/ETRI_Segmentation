#save_path: checkpoint path to save
#weights: path of weight to start from
#backbone: name of backbone
#model_name: name of current version of model to save.
#top_k: number of classes to use
#merged: name of key for using pre-defined list of classes to merge ambiguous classes
#source_dataset: name of source dataset to use
#target_dataset: name of target dataset to use
#source_root: the directory path of source dataset
#target_root: the directory path of target dataset
#pretrained: the path of weight of pre-trained dpt
CUDA_VISIBLE_DEVICES=1 python ./main.py \
--save_path "./checkpoints" \
--weights "" \
--backbone "dpt-hybrid" \
--model_name "etri_semi" \
--top_k 6 \
--batch_size 4 \
--ignore_index -1 \
--experiment_name "etri_semi" \
--merged "etri_merge_top6" \
--optimizer 'SGD' \
--init_lr 1e-4 \
--lr_max_iter 40000 \
--poly_power 0.9 \
--weight_decay 0.0001 \
--ema_decay 0.999 \
--source_dataset 'etri' \
--target_dataset 'DRAM' \
--source_root './datasets/etri' \
--target_root './datasets/DRAM_processed/' \
--pretrained "./dpt_hybrid-ade20k-53898607.pt" \
--lam_style 1.0 \
--pseudolabel_threshold 0.8 \
--lam_randaug 0.8 \
--lam_styleaug 0.8 \

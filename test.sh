CUDA_VISIBLE_DEVICES=0 python ./test.py \
--save_path "./checkpoints"\
--weights "./checkpoint/top6/source/Epoch::43:: | Model: etri_merge_top6  MIoU: 0.724 | MPA: 0.835083.pth" \
--model_name "Segmento-ADE20k" \
--top_k 6 \
--batch_size 1 \
--experiment_name "etri_test" \
--backbone "dpt-hybrid" \
--source_root './dataset/etri' \
--merged "etri_merge_top6" \

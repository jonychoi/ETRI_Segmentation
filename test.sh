CUDA_VISIBLE_DEVICES=1 python /home/cvlab02/project/etri/test.py \
--save_path "/media/dataset2/etri_final" \
--weights "/media/dataset2/etri_result/etri_semi/etri_semi/top6/source/Epoch::43:: | Model: etri_merge_top6  MIoU: 0.724 | MPA: 0.835083.pth" \
--model_name "Segmento-ADE20k" \
--top_k 6 \
--batch_size 1 \
--experiment_name "etri_test" \
--backbone "dpt-hybrid" \
--lr 5e-4 \
--source_root '/media/dataset2/etri' \
--merged "etri_merge_top6" \
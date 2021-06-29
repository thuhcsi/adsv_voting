#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
voxceleb1_path=~/datasets/VoxCeleb/voxceleb1
checkpoint_path=ckpt.pt

stage=5

if [ $stage -eq 1 ];then
	rm -rf data; mkdir data
	wget -P data/ https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
	echo format trails
	python3 scripts/format_trials.py \
		--voxceleb1_root $voxceleb1_path \
		--src_trials_path data/veri_test.txt \
		--dst_trials_path data/vox1.txt
fi

if [ $stage -eq 2 ];then
	python3 tools/evaluate.py \
		--config config/voting.yaml \
		--trial_path data/adv_data/adv_trials.lst \
		--checkpoint_path $checkpoint_path
fi

if [ $stage -eq 3 ];then
	python3 local/attack.py \
		--config config/voting.yaml \
		--trial_path data/vox1.txt \
		--checkpoint_path $checkpoint_path \
		--alpha 3.0 \
		--num_iters 5 \
		--epsilon 15 \
		--adv_save_dir data/adv_3_5
fi

if [ $stage -eq 4 ];then
	python3 tools/evaluate.py \
		--config config/voting.yaml \
		--trial_path data/adv_3_5/adv_trials.txt \
		--checkpoint_path $checkpoint_path
fi

if [ $stage -eq 5 ];then
	python3 local/defense.py \
		--config config/voting.yaml \
		--trial_path data/vox1.txt \
		--checkpoint_path $checkpoint_path \
		--trial_path data/adv_3_5/adv_trials.txt \
		--score_save_path data/adv_3_5/score.txt \
		--epsilon 30 \
		--num_voting 20
fi


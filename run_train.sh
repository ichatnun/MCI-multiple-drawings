#!/bin/bash

python main.py --random_seed=777 --include_clock --label_type="hard" --exp_name="clock_vgg16_hard_seed777" --model_name="vgg16" --num_epochs=100 --use_pretrained_weight 

python main.py --random_seed=777 --include_clock --label_type="soft" --exp_name="clock_vgg16_soft_seed777" --model_name="vgg16" --num_epochs=100 --use_pretrained_weight 
#!/bin/bash

# python main.py --num_seeds=10 --include_clock --label_type="hard" --model_name="vgg16" --num_epochs=100 --use_pretrained_weight 

# python main.py --num_seeds=10 --include_clock --label_type="soft" --model_name="vgg16" --num_epochs=100 --use_pretrained_weight 

python main.py --num_seeds=10 --include_clock --label_type="hard" --model_name="conv-att" --num_epochs=100 --use_pretrained_weight 

python main.py --num_seeds=10 --include_clock --label_type="soft" --model_name="conv-att" --num_epochs=100 --use_pretrained_weight

# python main.py --num_seeds=10 --include_clock --include_copy --include_trail --label_type="hard" --model_name="vgg16" --num_epochs=100 --use_pretrained_weight 

# python main.py --num_seeds=10 --include_clock --include_copy --include_trail --label_type="soft" --model_name="vgg16" --num_epochs=100 --use_pretrained_weight 

python main.py --num_seeds=10 --include_clock --include_copy --include_trail --label_type="hard" --model_name="conv-att" --num_epochs=100 --use_pretrained_weight 

python main.py --num_seeds=10 --include_clock --include_copy --include_trail --label_type="soft" --model_name="conv-att" --num_epochs=100 --use_pretrained_weight
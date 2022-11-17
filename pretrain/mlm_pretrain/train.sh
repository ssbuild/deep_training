
#--model_name_or_path /data/nlp/pre_models/torch/bert/bert-base-chinese \
#--max_epochs 5 \
#--max_steps 100000 \
python ../train.py \
--device '0' \
--data_backend 'leveldb' \
--model_type bert \
--model_name_or_path /data/nlp/pre_models/torch/bert/bert-base-chinese \
--tokenizer_name /data/nlp/pre_models/torch/bert/bert-base-chinese \
--config_name /data/nlp/pre_models/torch/bert/bert-base-chinese/config.json \
--train_file "/data/nlp/nlp_train_data/thucnews/train.json" \
--do_train true \
--max_steps 100000 \
--learning_rate 5e-5 \
--train_batch_size 8 \
--test_batch_size 2 \
--adam_epsilon 1e-8 \
--gradient_accumulation_steps 10 \
--max_grad_norm 1.0 \
--weight_decay 0 \
--warmup_steps 0 \
--output_dir './output' \
--do_lower_case=false \
--do_whole_word_mask=true \
--max_seq_length=512 \
--max_predictions_per_seq=20 \
--dupe_factor=5 \
--masked_lm_prob=0.15



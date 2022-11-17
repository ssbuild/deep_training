

python ../train.py \
--device '0' \
--data_backend 'leveldb' \
--model_type bert \
--model_name_or_path /data/nlp/pre_models/torch/bert/bert-base-chinese \
--tokenizer_name /data/nlp/pre_models/torch/bert/bert-base-chinese \
--config_name /data/nlp/pre_models/torch/bert/bert-base-chinese/config.json \
--do_train \
--train_file /data/nlp/nlp_train_data/clue/afqmc_public/train.json \
--eval_file /data/nlp/nlp_train_data/clue/afqmc_public/dev.json \
--test_file /data/nlp/nlp_train_data/clue/afqmc_public/test.json \
--learning_rate 5e-5 \
--max_epochs 3 \
--train_batch_size 64 \
--test_batch_size 2 \
--adam_epsilon 1e-8 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0 \
--warmup_steps 0 \
--output_dir './output' \
--max_seq_length 140



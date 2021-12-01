python my_main.py \
--bert_model \
../model/bert-large-uncased.tar.gz \
--train_batch_size 32 \
--output_dir EXP/ \
--learning_rate 1e-5 --num_train_epochs 24 \
--train_data \
../data/train.pt \
--eval_data \
../data/valid.pt \
--do_train \
--num_log_steps 100 \
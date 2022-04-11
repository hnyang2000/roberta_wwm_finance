# roberta_wwm_finance

使用哈工大roberta-wwm模型做字嵌入的BiLSTM-CRF序列标注模型


运行代码

!nvidia-smi<br />
\# !git clone https://github.com/hnyang2000/roberta_wwm_rmrb.git<br />
!pip install seqeval<br />
!pip install datasets<br />
!pip install wandb<br />
!pip install sacremoses<br />
!pip install tokenizers<br />
\# !wandb login<br />
!pip install ltp<br />
!pip install --upgrade huggingface-hub<br />

\# finance<br />
\# no dropout<br />
!mkdir /root/roberta-wwm-clue-best<br />
!cp /root/autodl-tmp/roberta-wwm-clue/config.json /root/autodl-tmp/roberta-wwm-clue/pytorch_model.bin /root/autodl-tmp/roberta-wwm-clue/special_tokens_map.json /root/autodl-tmp/roberta-wwm-clue/tokenizer.json /root/autodl-tmp/roberta-wwm-clue/tokenizer_config.json /root/autodl-tmp/roberta-wwm-clue/vocab.txt /root/roberta-wwm-clue-best<br />
\# --output_dir /root/autodl-tmp/roberta-wwm-finance --train_file finance_train.txt --validation_file finance_val.txt --test_file finance_test.txt --do_train True --do_eval True --do_predict False --per_device_train_batch_size 20 --per_device_eval_batch_size 96 --num_train_epochs 32 --resume_from_checkpoint /root/roberta-wwm-clue-best --evaluation_strategy epoch --load_best_model_at_end True --max_seq_length 256 --learning_rate 2e-5 --save_total_limit 25 --weight_decay 0.01 --max_grad_norm 5 --warmup_ratio 0.1<br />


\# inference<br />
--output_dir /root/autodl-tmp/roberta-wwm-clue --train_file finance_train.txt --validation_file finance_val.txt --test_file finance_test.txt --do_train False --do_eval False --do_predict True --per_device_train_batch_size 40 --per_device_eval_batch_size 64 --num_train_epochs 32 --model_name_or_path /root/roberta-wwm/roberta-wwm-clue --evaluation_strategy no --load_best_model_at_end False --save_total_limit 25<br />


\# mlm train<br />
--output_dir /root/autodl-tmp/mlm --train_file mlm_train.txt --validation_file mlm_val.txt --do_train True --do_eval True --do_predict False --per_device_train_batch_size 12 --per_device_eval_batch_size 64 --num_train_epochs 32 --model_name_or_path /root/roberta-wwm/roberta-wwm-clue --evaluation_strategy epoch --load_best_model_at_end True --max_seq_length 256 --learning_rate 2e-5 --save_total_limit 25<br />

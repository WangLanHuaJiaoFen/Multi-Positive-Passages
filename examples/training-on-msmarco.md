# MS MARCO passage training example
In this doc, we show the steps to train dense retriever on MS MARCO passage using SumMargLH.

## ⚠⚠⚠ Notice

Remember to adjust the dataset loading code in [src/tevatron/datasets/dataset.py](https://github.com/WangLanHuaJiaoFen/Multi-Positive-Passages/blob/main/src/tevatron/datasets/dataset.py) to fit your environment.

Besides, we use `load_from_disk` for locally saved datasets in the loading script instead of the universal `load_dataset` call, which allows faster loading from disk. This means you will need to modify the script to match your own dataset paths and storage format.
## Training
Noting that our datasets can be found [Wanglanhuajiaofen/MSMARCO-annotation](https://huggingface.co/datasets/Wanglanhuajiaofen/MSMARCO-annotation).

Refer to [src/tevatron/modeling/encoder.py](https://github.com/WangLanHuaJiaoFen/Multi-Positive-Passages/blob/main/src/tevatron/modeling/encoder.py) and [src/tevatron/arguments.py](https://github.com/WangLanHuaJiaoFen/Multi-Positive-Passages/blob/main/src/tevatron/arguments.py) to find how to use other losses.
```bash
nohup torchrun --nproc_per_node=2 -m tevatron.driver.train \
--output_dir your-output-dir \
--model_name_or_path your-model-dir \
--save_steps 3131 \
--dataset_name dataset-dir \
--fp16 \
--per_device_train_batch_size 64 \
--train_n_passages 8 \
--learning_rate 3e-5 \
--q_max_len 16 \
--p_max_len 128 \
--num_train_epochs 3 \
--logging_steps 500 \
--negatives_x_device \
--overwrite_output_dir \
--positive_type multiple \
--loss_type sum_mar \
--max_positive_num 4 \
--dataloader_num_workers 8 > your-log-dir 2>&1 &
```

## Encoding
### Encode query
```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
--output_dir temp \
--model_name_or_path your-output-dir-in-last-step \
--fp16 \
--per_device_eval_batch_size 156 \
--dataset_name dataset-dir \
--encoded_save_path query-emb-output-dir/query-emb.pkl \
--q_max_len 32 \
--encode_is_qry
```

### Encode Corpus
```bash
for s in $(seq -f "%02g" 0 19); do CUDA_VISIBLE_DEVICES=0  python -m tevatron.driver.encode \
--output_dir=temp \
--model_name_or_path your-output-dir-in-last-step \
--fp16 \
--per_device_eval_batch_size 512 \
--p_max_len 128 \
--dataset_name dataset-dir \
--encoded_save_path corpus-emb-output-dir/corpus_emp.${s}.pkl \
--encode_num_shard 20 \
--encode_shard_index ${s}; done
```

## Retrieval
Rely on faiss-gpu. Remember to run ``pip install faiss-gpu``. Do not install faiss-cpu.
```bash
for s in $(seq -f "%02g" 0 0); do CUDA_VISIBLE_DEVICES=0 python -m tevatron.faiss_retriever  \
--query_reps query-emb-output-dir/query-emb.pkl \
--passage_reps corpus-emb-output-dir/corpus_emp.${s}.pkl \
--depth 1000 \
--save_ranking_to result-dir/${s} \
--use_gpu; done
```

## Combining the results
```bash
python -m tevatron.faiss_retriever.reducer \
--score_dir result-dir/ \
--query query-emb-output-dir/query-emb.pkl \
--save_ranking_to final-result-dir/result.txt
```

## Evaluation
Convert result to MARCO format.
```bash
python -m tevatron.utils.format.convert_result_to_trec \
--input final-result-dir/result.txt \
--output final-result-dir/result.trec
```

Then evaluate using **pytrec-eval**.
```bash
python trec_eval.py  \
--qrel_file qrels.txt \
--run_file final-result-dir/result.trec
```
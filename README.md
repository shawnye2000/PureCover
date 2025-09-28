# PureCover 
This is the repository for the Paper "PureCover: Bridging the Gap in Re-ranking for Retrieval-Augmented Generation via Balancing Coverage and Noise".

## Environment
```
pip install vllm==0.6.0
pip install flashrag-dev[full]
# install faiss
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Create Index
```
CUDA_VISIBLE_DEVICES=2 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path xxx/.cache/huggingface/hub/models--intfloat--e5-base-v2/snapshots/1c644c92ad3ba1efdad3f1451a637716616a20e8/ \
    --corpus_path indexes/retrieval-corpus/wiki-18.jsonl \
    --save_dir indexes/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat
```



## Get Training dataset
```
python run_reranker.py --dataset hotpotqa --split train --teacher qwen2.5-32b --output_training_dataset_path xxx --api_model xxx --api_key xxx --api_base xxx ## the api here is for calling the reasoning of teacher model. 

```

## Set-wise Distillation
```
python distillation.py --output_training_dataset_path xxx --student_path your_student_model_path --save_model_path saved_student_model_path
```


## Inference
```
python run_reranker.py --dataset hotpotqa --split dev --student_path your_student_model_path --api_model xxx --api_key xxx --api_base xxx
```






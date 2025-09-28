# PureCover 
This is the repository for the Paper "PureCover: Bridging the Gap in Re-ranking for Retrieval-Augmented Generation via Balancing Coverage and Noise".

## Environment
```
pip install -r requirements.txt
```
Download your corpus data from [Huggingface](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus).
Download your dataset data from [Huggingface](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main) and save it in `dataset/`






## Create Index
```
CUDA_VISIBLE_DEVICES=2 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path your_e5_model_path \
    --corpus_path indexes/retrieval-corpus/wiki-18.jsonl \
    --save_dir indexes/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat
```



## Get Training dataset
Load your teacher model using the vllm api:
```
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server
        --model your_teacher_model_path
        --tensor-parallel-size 2
        --api-key EMPTY
        --port 8000 --dtype half
        --gpu-memory-utilization 0.9
```
run the training code
```
python run_reranker.py
        --dataset hotpotqa
        --split train
        --teacher qwen2.5-32b
        --output_training_dataset_path output_path
        --api_model your_teacher_model_path
        --api_key EMPTY
        --api_base http://localhost:8000/v1
```

## Set-wise Distillation
```
python distillation.py
        --output_training_dataset_path xxx
        --student_path your_student_model_path
        --save_model_path saved_student_model_path
```


## Inference
Load your LLM generator model using the vllm api:
```
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server
        --model your_teacher_model_path
        --tensor-parallel-size 2
        --api-key EMPTY
        --port 8000 --dtype half
        --gpu-memory-utilization 0.9
```
```
python run_reranker.py
        --dataset hotpotqa
        --split dev
        --student_path your_student_model_path
        --topk 5
        --api_model your_generator_path
        --api_key EMPTY
        --api_base http://localhost:8000/v1
```






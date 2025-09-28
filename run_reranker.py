import argparse
from flashrag.config import RerankConfig
from flashrag.utils import get_dataset
from flashrag.pipeline import RerankerPipeline
from flashrag.prompt import PromptTemplate

# from flashrag.utils import get_retriever, get_generator
import wandb
dataset_dict = {'hotpotqa': 'dev',
                'strategyqa': 'dev',
                '2wikimultihopqa': 'dev',
                'musique': 'dev',
                'triviaqa': 'test',
                'nq': 'test',}

if __name__ == '__main__':
    import argparse
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--dataset_name', type=str, default='hotpotqa')
    base_args.add_argument('--api_model', type=str, default="...")
    base_args.add_argument('--api_key', type=str,
                           default="EMPTY")
    base_args.add_argument('--api_base', type=str,
                           default="http://localhost:8000/v1")
    base_args.add_argument('--split', type=str,
                           default="train")
    base_args.add_argument('--lbd', type=float, default=0.3)
    base_args.add_argument('--tau', type=float, default=1e-5)
    base_args.add_argument('--topk', type=int, default=5)
    base_args.add_argument('--retriever', type=str, default='e5')
    base_args.add_argument('--teacher', type=str, default='qwen2.5-32b')
    base_args.add_argument('--student_path', type=str, default='...')
    base_args.add_argument('--output_training_dataset_path', type=str)
    args = base_args.parse_args()
    config_dict = {
        "dataset_name": args.dataset_name,
        "split": [args.split],
        "framework": "openai",
        "generator_model": args.api_model,
        "openai_setting": {"api_key": args.api_key,
                           "base_url": args.api_base},
        "retrieval_method": args.retriever, #
        "metrics": ["em", "f1", "acc", "bleu", "rouge-l", "retrieval_recall", "retrieval_precision", "input_tokens"],
        "metric_setting": {"retrieval_recall_topk": args.topk},
        "save_intermediate_data": True,
        "test_sample_num": 500,
        "faiss_gpu":  False,
        "rerank_method": 'purecover',
        'rerank_model_name': 'purecover',
        "device": "cuda:",
        'use_reranker': True,
        'retrieval_topk': 20,
    }

    if config_dict["split"][0] == 'train':
        config_dict.update({
            'rerank_topk': 20,
            'teacher_model': args.teacher,
            'output_training_dataset_path': '',
            'tau': args.tau,
            'lbd': args.lbd,
        })
    else:
        config_dict.update({
            'rerank_topk': args.topk, #5
            'student_path': args.student_path,
        })

    print(f'config_dict: {config_dict}')

    config = RerankConfig(config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split[config['split'][0]]
    prompt_templete = PromptTemplate(
        config,
        system_prompt="Answer the question based on the given document. \
                        Only give me the answer and do not output any other words. \
                        \nThe following are given documents.\n\n{reference}",
        user_prompt="Question: {question}\nAnswer:",
    )
    wandb.init(project="reranker", name=f"{config['rerank_method']}_reranker",
                 config=config_dict)
    pipeline = RerankerPipeline(config, prompt_template=prompt_templete)
    output_dataset = pipeline.run(test_data, do_eval=True)


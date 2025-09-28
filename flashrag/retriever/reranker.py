import random
from typing import List
import torch
import warnings
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from torch.optim import AdamW
from functools import partial
import math
from flashrag.retriever.encoder import Encoder
import torch.nn.functional as F
from flashrag.utils import (call_openai_chat,split_by_steps,
                            extract_str_to_list, get_last_list_str,  max_coverage_rerank, max_coverage_rerank_with_varing_k,
                            submodular_greedy_soft_coverage_and_max,
submodular_greedy_soft_coverage_and_noise,
denoised_submodular_greedy_soft_coverage,
                            get_sub_query, save_json, load_json, append_to_jsonl_file,
                            load_jsonl)
import wandb
import os

# from Test.cross_attention import context_length
from .qwen3_listwise_reranker_model import Qwen3ListwiseRerankerInferenceModel
from .qwen3_setwise_reranker_model2 import Qwen3SetwiseRerankerInferenceModel


class BaseReranker:
    r"""Base object for all rerankers."""

    def __init__(self, config):
        self.config = config
        self.reranker_model_name = config["rerank_model_name"]
        self.reranker_model_path = config["rerank_model_path"]
        self.topk = config["rerank_topk"]
        self.max_length = config["rerank_max_length"]
        self.batch_size = config["rerank_batch_size"]
        self.device = config["device"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["metric_setting"]["tokenizer_name"])

    def get_rerank_scores(self, query_list: List[str], doc_list: List[str], batch_size):
        """Return flatten list of scores for each (query,doc) pair
        Args:
            query_list: List of N queries
            doc_list:  Nested list of length N, each element corresponds to K documents of a query

        Return:
            [score(q1,d1), score(q1,d2),... score(q2,d1),...]
        """
        all_scores = []
        return all_scores

    @torch.inference_mode(mode=True)
    def rerank(self, query_list, doc_list, batch_size=None, topk=None):
        r"""Rerank doc_list."""
        if batch_size is None:
            batch_size = self.batch_size
        if topk is None:
            topk = self.topk
        if isinstance(query_list, str):
            query_list = [query_list]
        if isinstance(doc_list, np.ndarray):
            doc_list = doc_list.tolist()
        if not isinstance(doc_list[0], list):
            doc_list = [doc_list]

        print(f'length of query_list: {len(query_list)}')
        print(f'length of doc_list: {len(doc_list)}')
        # print('doc_list: ', doc_list)
        assert len(query_list) == len(doc_list)

        if topk < min([len(docs) for docs in doc_list]):
            warnings.warn("The number of doc returned by the retriever is less than the topk.")

        # get doc contents
        doc_contents = []
        for docs in doc_list:
            if all([isinstance(doc, str) for doc in docs]):
                doc_contents.append([doc for doc in docs])
            else:
                doc_contents.append([doc["contents"] for doc in docs])

        all_scores = self.get_rerank_scores(query_list, doc_contents, batch_size)

        # print(f"all_scores: {all_scores} \n length: {len(all_scores)} \n doc len:{sum([len(docs) for docs in doc_list])}")
        assert len(all_scores) == sum([len(docs) for docs in doc_list])

        # sort docs
        start_idx = 0
        final_scores = []
        final_docs = []

        for docs in doc_list:
            doc_scores = all_scores[start_idx : start_idx + len(docs)]
            doc_scores = [float(score) for score in doc_scores]
            start_idx += len(docs)
            # 只保留 score > 0 的项
            if self.config["rerank_method"] in ['cover', 'attn', 'setr']:
                filtered = [(doc, score) for doc, score in zip(docs, doc_scores) if score > 0]
                if filtered:
                    # 拆开文档与分数
                    filtered_docs, filtered_scores = zip(*filtered)

                    # 对 scores 排序并取 topk
                    sort_idxs = np.argsort(filtered_scores)[::-1][:topk]
                    print(f'sorted idxs: {sort_idxs}')
                    final_docs.append([filtered_docs[i] for i in sort_idxs])
                    final_scores.append([filtered_scores[i] for i in sort_idxs])
                else:
                    raise Exception
            else:
                sort_idxs = np.argsort(doc_scores)[::-1][:topk]
                #
                final_docs.append([docs[idx] for idx in sort_idxs])
                final_scores.append([doc_scores[idx] for idx in sort_idxs])

        token_counts = 0
        doc_counts = 0
        for docs_per_query in final_docs:
            for doc in docs_per_query:
                token_counts += len(self.tokenizer.tokenize(doc["contents"]))
            doc_counts += len(docs_per_query)
        print(f'total tokens: {token_counts}')
        print(f'token_count per query: {token_counts/len(final_docs)}')
        print(f'total docs: {doc_counts}')
        print(f'doc_count per query: {doc_counts/len(final_docs)}')

        return final_docs, final_scores


class CrossReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_path)
        self.ranker = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_path, num_labels=1)
        print(f'self.model_path: {self.reranker_model_path}')
        # self.ranker = torch.nn.DataParallel(self.ranker, device_ids=[0, 1])
        self.ranker.to(self.device)
        self.ranker.eval()

    @torch.inference_mode(mode=True)
    def get_rerank_scores(self, query_list, doc_list, batch_size):
        # flatten all pairs
        all_pairs = []
        for query, docs in zip(query_list, doc_list):
            all_pairs.extend([[query, doc] for doc in docs])
        all_scores = []
        for start_idx in tqdm(range(0, len(all_pairs), batch_size), desc="Reranking process: "):
            pair_batch = all_pairs[start_idx : start_idx + batch_size]

            inputs = self.tokenizer(
                pair_batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
            ).to(self.device)
            # print(f'inputs: {inputs}')
            batch_scores = (
                self.ranker(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
                .cpu()
            )
            all_scores.extend(batch_scores)

        return all_scores


class AttentionReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.teacher_model = self.config['teacher_model']
        self.teacher_model_path = self.config['model2path'][self.teacher_model]
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_path)
        self.tokenizer.padding_side = "left"
        self.dataset_path = os.path.join(self.config['data_dir'], self.config["dataset_name"], f'{self.config["split"][0]}.jsonl')
        self.dataset = load_jsonl(self.dataset_path)
        self.query2answer = {item["question"]: item["golden_answers"][0] for item in self.dataset}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.training_dataset_path = self.config['output_training_dataset_path']
        self.tau = self.config['tau']
        self.lbd = self.config['lbd']

    def choose_model(self):
        if self.teacher_model == 'llama3':
            from .llama3d1_hook_attn import LlamaForCausalLM
            self.model = LlamaForCausalLM.from_pretrained(
                self.teacher_model_path,
                torch_dtype=torch.bfloat16, device_map="auto",
                _attn_implementation='flash_attention_2',
                use_cache=False
            )
        elif self.teacher_model == 'qwen2-7b':
            from .qwen2d1_hook_attn import Qwen2ForCausalLM
            self.model = Qwen2ForCausalLM.from_pretrained(
                self.teacher_model_path,
                torch_dtype=torch.bfloat16, device_map="auto",
                _attn_implementation='flash_attention_2',
                use_cache=False
            )
        elif self.teacher_model == 'qwen2.5-32b':
            from .qwen2d1_hook_attn import Qwen2ForCausalLM
            self.model = Qwen2ForCausalLM.from_pretrained(
                self.teacher_model_path,
                torch_dtype=torch.bfloat16, device_map="auto",
                _attn_implementation='flash_attention_2',
                use_cache=False
            )
        else:
            raise NotImplementedError

    @torch.inference_mode(mode=True)
    def get_position_bias_weight(self, query_list, doc_list):
        dummy_doc = (
            "This paragraph is completely unrelated to any query and contains "
            "generic content to estimate positional attention bias. It includes "
            "neutral factual statements: “Paris is the capital of France.”, “Water "
            "freezes at 0°C.”, “John works as a software engineer.”, “The sky is blue.”, "
            "“Mount Everest is the highest mountain on Earth.” By repeating generic "
            "sentences, we ensure the dummy content provides no useful information "
            "other than fixed context length and token positions. The purpose of this "
            "text is solely to maintain a consistent context structure, with no "
            "semantic link to the query. This repetition is intentional, ensuring "
            "uniformity across all positions during bias measurement."
        )
        all_weight = []
        import copy
        for query, docs in tqdm(zip(query_list[:100], doc_list[:100]), total=len(query_list[:5]),desc="IIIII process: "):
            weight = []
            for k in trange(20):
                doc_w_dummy_docs = copy.deepcopy(docs)
                doc_w_dummy_docs[k] = dummy_doc[:600]
                A, correct = self.get_adj_matrix(query, doc_w_dummy_docs)
                weight_avg_step = np.mean(A[k, :])
                weight.append(weight_avg_step)
                print(f'weighted avg step: {weight_avg_step}')
            all_weight.append(weight)
            print(f'weight: {weight}')
            print(f'all_weight: {all_weight}')
        print(all_weight)
        all_weight = np.array(all_weight)
        np.savetxt(self.calibrated_path, all_weight, fmt="%.8f")
        return np.mean(all_weight, axis=0)

    def get_avg_attention_from_source_to_target(self, source, target, input_ids, avg_attn ):
        source_ids = self.get_indices(source, self.tokenizer, input_ids)
        source_ids = torch.tensor(source_ids, device=self.device)
        target_ids = self.get_indices(target, self.tokenizer, input_ids)
        target_ids = torch.tensor(target_ids, device=self.device)
        seq_len = avg_attn.shape[0]
        assert (source_ids >= 0).all() and (source_ids < seq_len).all(), \
            f"source_ids 超出范围: max {source_ids.max()}, seq_len {seq_len}"
        assert (target_ids >= 0).all() and (target_ids < seq_len).all(), \
            f"target_ids 超出范围: max {target_ids.max()}, seq_len {seq_len}"
        # print(f'source_ids: {source_ids} target_ids: {target_ids}')
        attn_for_doc = avg_attn[source_ids.long().unsqueeze(1), target_ids.long().unsqueeze(0)]  # 所有输出 token 对该 doc 的 token 的注意力
        attn_for_doc_avg = attn_for_doc.mean(dim=0)
        score = attn_for_doc_avg.mean().item()  # 先对 token_ids 平均，再对所有输出 token 平均
        return score

    @torch.inference_mode(mode=True)
    def get_avg_attn(self, input_ids, context_length, attention_mask):
        # attn_dict = {}

        hook = Hook()
        all_hooks = []
        for layer in self.model.model.layers:
            all_hooks.append(layer.self_attn.register_forward_hook(hook.hook_fn))

        self.layer_num = len(self.model.model.layers)
        print(f'layer Num: {self.layer_num}')
        with torch.no_grad():
            self.model(input_ids=input_ids)
            for h in all_hooks:
                h.remove()
            # 获取所有的keys和queries
            avg_attn_per_layer = []
            for i in range(self.layer_num):     ##, total=32):  # 遍历层
                queries = hook.queries[i]  # [bsz, 32(heads num), seq_len, head_dim]
                keys = hook.keys[i]  # [bsz, 32(heads num), seq_len, head_dim]
                # values = hook.values[i] # [bsz, 8(key heads num), seq_len, head_dim]
                # 计算不同头上的attention map
                n_heads = queries.shape[1]
                head_dim = queries.shape[-1]
                avg_attn_per_head = []
                for h in range(n_heads):
                    q = queries[0, h, :, :]  # [context_length:, head_dim]
                    k = keys[0, h, :, :]  # [seq_len, head_dim]

                    # 计算 Scaled Dot-Product Attention logits
                    attn = torch.matmul(q, k.T) / math.sqrt(head_dim)  # [question_len, context_len + question_len]
                    # print(f'attn: {attn.shape}')
                    # print(f'attn mask:{attention_mask.shape}')
                    # 加 causal mask（防止看到未来）
                    attn[context_length:, context_length:] += attention_mask[0, 0, :, :].to(attn.device)  # [question_len, context_len + question_len]下三角梯形

                    # 归一化为概率分布
                    attn_weights = F.softmax(attn, dim=-1).to(self.device)  # [seq_len, seq_len]
                    # print(f"Layer {i}, Head {h}:\n Attention shape: {attn_weights.shape}")
                    # print(f"Layer {i}, Head {h}:\n Attention values: {attn_weights}")
                    # attn_dict[(i, h)] = attn_weights
                    avg_attn_per_head.append(attn_weights)
                avg_attn_per_head = torch.stack(avg_attn_per_head, dim=0)
                avg_attn_per_head = avg_attn_per_head.mean(dim=0).to(self.device)
                avg_attn_per_layer.append(avg_attn_per_head)
            avg_attn_per_layer = torch.stack(avg_attn_per_layer, dim=0)
            avg_attn = avg_attn_per_layer.mean(dim=0)
        return avg_attn

    def get_indices(self, text, tokenizer, input_ids):
        input_text = tokenizer.decode(input_ids[0])
        doc_token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
        doc_text = tokenizer.decode(doc_token_ids)
        start = input_text.find(doc_text)
        if start == -1:
            raise ValueError(f"Doc {text} not found in input.")

        start_token = tokenizer(input_text[:start], add_special_tokens=False).input_ids
        start_idx = len(start_token)
        end_idx = start_idx + len(doc_token_ids)

        seq_len = input_ids.shape[1]
        # 防止越界
        end_idx = min(end_idx, seq_len)

        return list(range(start_idx, end_idx))


    def process_input(self, input_str, context_str):
        inputs = self.tokenizer(input_str,
                                return_tensors="pt",
                                add_special_tokens=True).to(self.device)
        input_ids = inputs["input_ids"]
        context_ids = self.tokenizer(context_str, return_tensors="pt", add_special_tokens=True).to(self.device)
        context_length = context_ids["input_ids"].shape[1]
        attention_mask = (
                (1 - torch.tril(torch.ones(len(input_ids[0, context_length:]),
                                           len(input_ids[0, context_length:]))))
                * torch.finfo(torch.bfloat16).min
        ).unsqueeze(0).unsqueeze(0).cpu()
        return input_ids, context_length, attention_mask

    @torch.inference_mode(mode=True)
    def get_adj_matrix(self, query, docs, position_bias_weight=None, mode='test'):
        print(f'position_bias_weight: {position_bias_weight}')
        prompt = ""
        for i, doc in enumerate(docs):
            prompt = prompt + f"[Doc {i}]" + doc + '\n'
        prompt = prompt + '\nQuery:' + query
        cot_example = (
                "\nYou are given a query and a set of documents."
                "Your goal is to reason through the documents step by step to answer the query.\n" # as efficiently as possible
                "Instructions:\n" 
                "1.	Only include reasoning steps that clearly contribute to solving the query.\n" 
                "2. Make sure each step is logically connected and leads to the final answer.\n"
                "3.	Use specific evidence from the documents in each step (e.g., facts, names, dates, relationships).\n"
                "4. In each step, cite document snippets or facts that directly support your reasoning."
                "5.	Avoid vague or redundant steps. Do not repeat the query.\n"
                "5.	Limit reasoning to 5 steps or fewer. Be concise and precise.\n"
                "Format:\n"
                "Each step must start with [Step N]: followed by your reasoning.\n"
                "Conclude with [Answer]: followed by the final answer (and nothing else).\n"
                "Example format:\n"
                "[Step 1]: ...\n"
                "[Step 2]: ...\n"
                "...\n"
                "[Answer]: ..."
                )

        response = call_openai_chat(prompt=prompt + cot_example, model = self.config["generator_model"],
                                             base_url =self.config["openai_setting"]["base_url"],
                                             api_key = self.config["openai_setting"]["api_key"])
        cots = split_by_steps(response)
        prompt += '\nAnswer:'
        answer_text = cots[-1]
        gt_answer = self.query2answer[query]
        print(f'Query:{query} \n Answer: {answer_text}  \n Gt answer:{gt_answer} ')
        cot_prompt =  '\n'.join(cots)
        all_input_prompt = prompt + cot_prompt
        input_ids, context_length, attention_mask = self.process_input(all_input_prompt,
                                                                   context_str=prompt)
        # -----------------------------------
        # Get LLm attention
        avg_attn = self.get_avg_attn(input_ids, context_length, attention_mask)

        cots_wo_answer = cots[:-1] #
        A = np.zeros((len(docs), len(cots_wo_answer)))
        for cot_id, cot in enumerate(cots_wo_answer):
            doc_attn_scores = {}
            for doc_id, doc in enumerate(docs):
                score = self.get_avg_attention_from_source_to_target(source=cot,
                                                                     target=doc,
                                                                     input_ids=input_ids,
                                                                     avg_attn=avg_attn)
                doc_attn_scores[doc_id] = score
                print(f'cot_id:{cot_id} cot:{cot} \n doc_id: {doc_id} \n  score: {score} \n doc_content:{docs[doc_id]} ')

            scores = torch.tensor(list(doc_attn_scores.values()), device=self.device)
            if position_bias_weight is None or self.config["rerank_method"] == 'wo_cali':
                scores = scores
                A[:, cot_id] = scores.cpu().numpy()
            else:
                scores = scores - torch.tensor(position_bias_weight, device=self.device)
                softmax_scores = torch.softmax(torch.tensor(scores, device=self.device) / self.tau, dim=0)
                A[:, cot_id] = softmax_scores.cpu().numpy()

        a = np.ones(len(cots_wo_answer))
        for cot_id, cot in enumerate(cots_wo_answer):
            score = self.get_avg_attention_from_source_to_target(source=cots[-1],
                                                                 target=cot,
                                                                 input_ids=input_ids,
                                                                 avg_attn=avg_attn)
            a[cot_id] = score
        softmax_a = torch.softmax(torch.tensor(a, device=self.device) / 0.005, dim=0)
        # clean
        import gc
        del input_ids, attention_mask, avg_attn #inputs,, avg_attn_per_layer #att_per_layer, att_stack,
        torch.cuda.empty_cache()
        gc.collect()
        if gt_answer in answer_text:
            print(f' Answer correct')
            return A, True, softmax_a
        else: #mode == 'train':
        #     return -1
            print(f' Answer Incorrect')
            return A, False, softmax_a

    @torch.inference_mode(mode=True)
    def model_infer(self, query_list, doc_list):
        all_scores = []
        for query, docs in tqdm(zip(query_list, doc_list), total=len(query_list), desc="Inferening process: "):
            # all_pairs.extend([(query, docs)])
            print(f'query: {query}')
            pairs = [(query, docs)] #list(zip(query, docs))
            new_scores = self.ranker.predict(pairs)
            all_scores.extend(new_scores)

        self.ranker.stop()
        return all_scores


    def get_rerank_scores(self, query_list, doc_list, batch_size):
        self.calibrated_path = './position_bias_array.txt'
        if os.path.exists(self.calibrated_path):
            position_bias_weight = np.mean(np.loadtxt(self.calibrated_path), axis=0)
        else:
            position_bias_weight = self.get_position_bias_weight(query_list, doc_list)
        print(f'position_bias_weight: {position_bias_weight}')
        print(self.config["split"])
        if self.config["split"][0] == 'train':
            data = list(zip(query_list, doc_list))
            if os.path.exists(self.training_dataset_path):
                exist_data_len = len(load_jsonl(self.training_dataset_path))
                print(f"训练数据集共有 {exist_data_len} 条样本")
                data = data[exist_data_len:]
            query_list, doc_list = zip(*data)
            query_list = list(query_list)
            doc_list = list(doc_list)
            for query, docs in tqdm(zip(query_list, doc_list), total=len(query_list), desc="Training process: "):
                A, correct, a = self.get_adj_matrix(query, docs, position_bias_weight=position_bias_weight, mode="train")
                if not isinstance(A, np.ndarray):
                    continue
                select_docids, select_docscores, _ = submodular_greedy_soft_coverage_and_noise(A, K=self.topk, threshold=0.0, lmbda=self.lbd)

                if self.training_dataset_path is not None:
                    print(f'select docids: {select_docids}')
                    new_data = {'query': query,
                                'candidates': docs,
                                'ranking_scores': select_docscores,
                                'ranking': select_docids,
                                'correct': correct}
                    append_to_jsonl_file(self.training_dataset_path,
                                         new_data=new_data)
                    print('saved.')
        else:
            self.ranker = Qwen3SetwiseRerankerInferenceModel(self.config['student_path'], model_type='vllm')
            scores = self.model_infer(query_list, doc_list)
            scores = [1.0 if s > self.tau else 0.0 for s in scores]
            return scores                                                                 # lmbda=self.lbd max_coverage_rerank_with_varing_k(A, K=self.topk)

class Hook():
    def __init__(self):
        self.keys = []
        self.queries = []
        self.values = []


    def hook_fn(self, module, input, output): #
        qkv = output[1]
        self.queries.append(qkv[1].transpose(1,2).detach())
        self.keys.append(qkv[2].detach())
        self.values.append(qkv[3].detach())
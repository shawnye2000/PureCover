import logging

import json
import logging
import wandb
import os
from peft import get_peft_model, LoraConfig, TaskType
from scipy.special.cython_special import kl_div
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset12 import cross_entropy_loss
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoModel, is_torch_npu_available,  get_scheduler)
logger = logging.getLogger(__name__)
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
import math
from flashrag.utils import load_jsonl
import wandb
from sentence_transformers import CrossEncoder, SentenceTransformer
from vllm.inputs.data import TokensPrompt
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from tqdm import trange, tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random

def format_instruction_transformer(instruction, query, docs):
    document_prompt = ""
    for i, doc in enumerate(docs):
        letter = chr(ord('A') + i)
        document_prompt = document_prompt + f"Passage {letter}:" + doc + "\n"
    output = f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>:{document_prompt}"
    return output

def top5_docs_to_letters(doc_scores):
    # 取分数最高的 5 个索引
    top_indices = sorted(range(len(doc_scores)), key=lambda i: -doc_scores[i])[:5]
    # 按分数顺序转换为字母
    return ','.join(chr(ord('A') + i) for i in top_indices)


def shuffle_string(s: str) -> str:
    chars = list(s)
    random.shuffle(chars)
    return ''.join(chars)


from scipy.optimize import linear_sum_assignment
def set_matching_loss(pred_logits, gold_token_ids):
    """
    Args:
        pred_logits: (num_pred, vocab_size)
        gold_token_ids: (num_gold, )
    """
    num_pred, vocab_size = pred_logits.shape
    num_gold = gold_token_ids.shape[0]

    cost_matrix = torch.zeros((num_pred, num_gold), device=pred_logits.device)

    for i in range(num_pred):
        for j in range(num_gold):
            cost_matrix[i][j] = F.cross_entropy(pred_logits[i].unsqueeze(0), gold_token_ids[j].unsqueeze(0), reduction='none')

    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    matched_loss = cost_matrix[row_ind, col_ind].mean()
    return matched_loss





class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, prefix, suffix, instruction, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = prefix
        self.suffix = suffix
        self.instruction = instruction
        self.doc_num = 20
        self.topk = 5
        self.shift_N = 9 #12
        self.sample_N = 5 #6
        self.samples = []  # 这是扁平化后的最终样本列表
        raw_data = load_jsonl(file_path)
        for item in raw_data:
            query = item["query"]
            candidates = item["candidates"][:self.doc_num]
            # is_relevant = item["is_relevant"]
            correct = item["correct"]
            # if not correct:
            #     continue
            ranking = [r for r in item["ranking"] if r < self.doc_num]  # 排序的doc id从高到低
            ranking_scores = item['ranking_scores'] # doc id对应的score
            doc_id2score = {id: s for id, s in zip(ranking, ranking_scores)}
            doc_scores = [doc_id2score[id] for id in range(self.doc_num)]
            # 打包成元组并打乱
            paired = list(zip(candidates, doc_scores))
            random.shuffle(paired)
            # 解包
            candidates,  doc_scores = zip(*paired)
            candidates = list(candidates)
            doc_scores = list(doc_scores)

            input_text = format_instruction_transformer(self.instruction, query, candidates)
            input_text = prefix + input_text + suffix
            output_text = top5_docs_to_letters(doc_scores)
            print(f'output_text: {output_text}')
            print(f'doc_scores: {doc_scores}')
            # gt_tokens = [self.tokenizer(t, add_special_tokens=False).input_ids[0] for t in output_text]

            self.samples.append({'input_text': input_text,
                                 'output_text': output_text,
                                 'doc_scores': doc_scores})

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample['input_text']
        output_text = sample['output_text']
        doc_scores = sample['doc_scores']
        # print(f'Full text: {full_text}')
        full_tokenized = self.tokenizer(input_text + output_text,
                                   truncation=False,
                                   max_length=self.max_length,
                                   return_tensors="pt")
        full_input_ids = full_tokenized["input_ids"].squeeze(0)
        full_attention_mask = full_tokenized["attention_mask"].squeeze(0)

        full_labels = full_input_ids.clone()

        input_tokenized = self.tokenizer(input_text,
                                   truncation=False,
                                   max_length=self.max_length,
                                   return_tensors="pt")
        input_input_ids = input_tokenized["input_ids"].squeeze(0)

        full_labels[:len(input_input_ids)] = -100

        return {
            "full_input_ids": full_input_ids,
            "full_attention_mask": full_attention_mask,
            "full_labels": full_labels,
            "input_input_ids": input_input_ids,
            "output_text": output_text,
            "doc_scores": doc_scores,
        }

def collate_fn(batch):
    full_input_ids = [x["full_input_ids"] for x in batch]
    full_attention_mask = [x["full_attention_mask"] for x in batch]
    full_labels = [x["full_labels"] for x in batch]
    input_input_ids = [x["input_input_ids"] for x in batch]
    # input_attention_mask = [x["input_attention_mask"] for x in batch]
    # input_labels = [x["input_labels"] for x in batch]
    doc_scores = [x["doc_scores"] for x in batch]
    output_text = [x["output_text"] for x in batch]

    full_input_ids = torch.nn.utils.rnn.pad_sequence(full_input_ids, batch_first=True, padding_value=0)
    full_attention_mask = torch.nn.utils.rnn.pad_sequence(full_attention_mask, batch_first=True, padding_value=0)
    full_labels = torch.nn.utils.rnn.pad_sequence(full_labels, batch_first=True, padding_value=-100)
    input_input_ids = torch.nn.utils.rnn.pad_sequence(input_input_ids, batch_first=True, padding_value=0)
    # input_attention_mask = torch.nn.utils.rnn.pad_sequence(input_attention_mask, batch_first=True, padding_value=0)
    # input_labels = torch.nn.utils.rnn.pad_sequence(input_labels, batch_first=True, padding_value=-100)
    doc_scores = torch.tensor(doc_scores)
    return {
        "full_input_ids": full_input_ids,
        "full_attention_mask": full_attention_mask,
        "full_labels": full_labels,
        "input_input_ids": input_input_ids,
        # "input_attention_mask": input_attention_mask,
        # "input_labels": input_labels,
        "doc_scores": doc_scores,
        "output_text": output_text,
    }


class Qwen3SetwiseRerankerInferenceModel:
    def __init__(self, model_name, model_type='transformers',  instruction="Given the user query, select the relevant passages that covers all evidence while minimize noise.", **kwargs):
        # self.lm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32)
        number_of_gpu=torch.cuda.device_count()
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = kwargs.get('max_length', 8192)
        if model_type == 'vllm':
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.alphabetic_tokens = [self.tokenizer(identifier, add_special_tokens=False).input_ids[0] for identifier in map(chr, range(ord('A'), ord('Z') + 1))]
            # self.greator_token = self.tokenizer(">", add_special_tokens=False).input_ids[0]
            self.sampling_params = SamplingParams(
                temperature=0,
                top_p=0.95,
                max_tokens=3, #100
                logprobs=20,
                allowed_token_ids=self.alphabetic_tokens, #qwen3_setwise_reranker_model.py
            ) #self.true_token,self.false_token,
            self.lm = LLM(model=model_name, tensor_parallel_size=number_of_gpu, max_model_len=10000, enable_prefix_caching=True, distributed_executor_backend='ray', gpu_memory_utilization=0.2)

        elif model_type == 'transformers':
            self.tokenizer.padding_side = "left"
            self.lm = AutoModelForCausalLM.from_pretrained(model_name,
                                                           trust_remote_code=True,
                                                           attn_implementation='flash_attention_2',
                                                           torch_dtype=torch.float16,
                                                           use_cache=False).cuda()  # .eval() #, attn_implementation="flash_attention_2"
            self.lm = torch.nn.DataParallel(self.lm).cuda()
            self.lr = 5e-4
            self.batch_size = 2
            self.epochs = 10
            self.use_fp16 = False
            self.use_lora = False
            self.gradient_accumulation_steps = 1
            self.device = next(self.lm.parameters()).device
            self.alphabetic_tokens = [self.tokenizer(identifier, add_special_tokens=False).input_ids[0] for identifier
                                      in map(chr, range(ord('A'), ord('Z') + 1))]
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be the passage label (from A to Z).<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def prepare_lora_model(self):
        # 先加载原模型 self.lm
        # 这里假设self.lm是huggingface风格的transformers模型

        config = LoraConfig(
            r=4,  # LoRA rank，通常4-16之间
            lora_alpha=16,  # LoRA alpha，缩放参数
            target_modules=["q_proj", "v_proj"],  # 你要微调的模块，常选 attention 的q和v权重
            lora_dropout=0.05,  # Dropout
            bias="none",  # 是否训练bias参数，"none"表示不训练
            task_type=TaskType.CAUSAL_LM  # 根据任务改，这里是语言模型
        )

        self.lm = get_peft_model(self.lm, config)
        self.lm = torch.nn.DataParallel(self.lm).cuda()
        # 冻结原模型所有参数，只训练LoRA参数
        for name, param in self.lm.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def format_instruction_vllm(self, instruction, query, docs):
        if isinstance(query, tuple):
            instruction = query[0]
            query = query[1]

        document_prompt = ""
        for i, doc in enumerate(docs):
            letter = chr(ord('A') + i)
            document_prompt = document_prompt + f"Passage {letter}:" + doc + "\n"
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be the passage label (from A to Z)."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>:{document_prompt}"}
        ]
        return text

    def format_instruction_transformer(self, instruction, query, docs):
        if instruction is None:
            instruction = self.instruction
        output = format_instruction_transformer(instruction, query, docs)
        return output

    def process_batch(self, pairs, **kwargs):
        messages = [self.format_instruction_vllm(self.instruction, query, docs) for query, docs in pairs]
        doc_numbers = [len(docs) for _, docs in pairs]
        # print(f'mesages: {messages}')
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:self.max_length] + self.suffix_tokens for ele in messages]
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        outputs = self.lm.generate(messages, self.sampling_params, use_tqdm=False)
        # print(f'len(outputs) = {len(outputs)}')
        # print(f'outputs = {outputs}')
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[0] # -1
            token_count = len(outputs[i].outputs[0].token_ids)
            alphabet_logits = []
            for tok in self.alphabetic_tokens:
                if tok in final_logits:
                    print(f'token: {tok}, logit: {final_logits[tok].logprob}')
                    alphabet_logits.append(math.exp(final_logits[tok].logprob))
                else:
                    print(f'token: {tok}, logit: {-10}')
                    alphabet_logits.append(math.exp(-10))
            alphabet_logits = alphabet_logits[:doc_numbers[i]]
            sum_score = sum(alphabet_logits)
            score = [l/sum_score for l in alphabet_logits]
            scores.extend(score)
        return scores


    def train_SFT(self, dataset_path, save_path):
        dataset = SFTDataset(dataset_path, self.tokenizer,
                             prefix=self.prefix,
                             suffix=self.suffix,
                             instruction=self.instruction,
                             max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        # 创建 dataset 和 dataloader
        # optimizer = AdamW(self.lm.parameters(), lr=self.lr)
        if self.use_lora:
            self.prepare_lora_model()
            optimizer = SGD(filter(lambda p: p.requires_grad, self.lm.parameters()), lr=self.lr)
        else:
            optimizer = SGD(self.lm.parameters(), lr=self.lr)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100,
                                     num_training_steps=self.epochs * len(dataloader))
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        self.lm.train()
        for epoch in trange(self.epochs, desc="Training Epoch"):
            epoch_loss = 0.0
            for step, batch in enumerate(tqdm(dataloader)):
                input_input_ids = batch["input_input_ids"].to(self.device)
                full_input_ids = batch["full_input_ids"].to(self.device)
                full_attention_mask = batch["full_attention_mask"].to(self.device)
                full_labels = batch["full_labels"].to(self.device)
                doc_scores = batch["doc_scores"].to(self.device)
                ground_truth_text = batch["output_text"]
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    outputs = self.lm(input_ids=full_input_ids, attention_mask=full_attention_mask, labels=full_labels)
                    output_starts =  (input_input_ids != 0).sum(dim=1)  # shape: [batch_size]
                    print("pad_token_id:", self.tokenizer.pad_token_id)
                    # 从 output_start 开始，取后面所有 token 的预测（即 argmax）
                    end = min(output_starts[0] -1 + 3, outputs.logits.shape[1])
                    pred_token_ids = torch.argmax(outputs.logits[0, (output_starts[0] - 1):end, :],
                                                  dim=-1)  # 形状: [output_len]
                    # 解码成文本
                    predicted_text = self.tokenizer.decode(pred_token_ids, skip_special_tokens=True)
                    print(f"Predicted output text:{predicted_text}")
                    print(f'ground truth text: {ground_truth_text[0]}')


                    sft_loss = outputs.loss.mean()

                    # batch_scores = outputs.logits[:, output_start, :]
                    batch_size = full_input_ids.shape[0]
                    batch_scores = []

                    for i in range(batch_size):
                        start_idx =  output_starts[i].item() - 1
                        score = outputs.logits[i, start_idx, :]  # 第一个 output token 的 logits
                        batch_scores.append(score)

                    batch_scores = torch.stack(batch_scores)  # shape: [batch_size, vocab_size]
                    alphabet_logits = []
                    for tok in self.alphabetic_tokens[:20]:
                        alphabet_logits.append(batch_scores[:, tok])
                    batch_scores = torch.stack(alphabet_logits, dim=1)
                    print(f'batch_scores: {batch_scores}\n '
                          f'doc_scores: {doc_scores}')
                    bpr_loss = self.bpr_loss_sample(batch_scores, doc_scores)
                    total_loss = 0.01* sft_loss + bpr_loss
                    print(f'bpr_loss: {bpr_loss}\n',
                          f'sft_loss: {sft_loss}\n')


                wandb.log({"batch_loss": total_loss,
                           'ranknet_loss': ranknet_loss,
                           #"bpr_loss": bpr_loss,})
                           # "cross_entropy_loss": cross_entropy_loss,
                           # "js_loss": js_loss,
                            "sft_loss": sft_loss,})

                print(f'batch_loss: {total_loss}')
                scaler.scale(total_loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.lm.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                if (step + 1) % 500 == 0:
                    self.save_model(save_path)

                epoch_loss += total_loss.item()
            wandb.log({"epoch_loss": epoch_loss})
            print(f'epoch_loss: {epoch_loss}')

            self.save_model(f'...')

    def bpr_loss_sample(self, batch_scores, doc_scores):
        """
        batch_scores: [B, N] — logits for each of N candidates per sample
        doc_scores:   [B, N] — ground-truth scores for each of N candidates per sample
        """
        batch_size, num_candidates = batch_scores.shape
        total_loss = 0.0
        count = 0

        for i in range(batch_size):
            # 按 doc_scores 排序（降序）
            sorted_indices = torch.argsort(doc_scores[i], descending=True)

            # 前 5 个作为正样本
            pos_indices = sorted_indices[:5]
            pos_scores = batch_scores[i][pos_indices]

            # 最后 10 个作为负样本候选，采样 5 个
            neg_pool_indices = sorted_indices[-10:]
            if len(neg_pool_indices) == 0 or len(pos_indices) == 0:
                continue
            sampled_neg_indices = neg_pool_indices[torch.randperm(len(neg_pool_indices))[:5]]
            neg_scores = batch_scores[i][sampled_neg_indices]

            # BPR pairwise loss
            diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # [num_pos, num_neg]
            loss = -torch.log(torch.sigmoid(diff)).mean()

            total_loss += loss
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, requires_grad=True, device=batch_scores.device)

    def bpr_loss(self, batch_scores: torch.Tensor, doc_scores: torch.Tensor) -> torch.Tensor:
        """
        BPR loss for batch ranking.

        Args:
            batch_scores: Tensor of shape [B, N] — model predicted scores/logits
            doc_scores:   Tensor of shape [B, N] — ground-truth relevance scores

        Returns:
            Scalar tensor representing the average BPR loss across all positive-negative pairs.
        """
        B, N = batch_scores.shape
        loss_list = []

        for b in range(B):
            pred = batch_scores[b]  # [N]
            gt = doc_scores[b]  # [N]

            # Find all (i, j) pairs such that gt[i] > gt[j]
            pos_indices = []
            neg_indices = []

            for i in range(N):
                for j in range(N):
                    if gt[i] > gt[j]:
                        if gt[j] > 0.1:
                            continue
                        else:
                            pos_indices.append(i)
                            neg_indices.append(j)

            if len(pos_indices) == 0:
                continue  # skip if no pairs

            pos_scores = pred[pos_indices]
            neg_scores = pred[neg_indices]

            # BPR loss for this sample
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss_list.append(loss)

        if len(loss_list) == 0:
            return torch.tensor(0.0, requires_grad=True, device=batch_scores.device)

        return torch.stack(loss_list).mean()

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        if isinstance(self.lm, torch.nn.DataParallel):
            self.lm.module.save_pretrained(save_path)
        else:
            self.lm.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f'Saved model to path: {save_path}')

    def start(self):
        pass

    def predict(
        self,
        sentences: list[tuple[str, list[str]]] ,  # query  docs
        batch_size: int = None,
        show_progress_bar: bool | None = False,
        num_workers: int = 1,
        activation_fct = None,
        apply_softmax: bool | None = False,
        convert_to_numpy: bool =  True,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> list[torch.Tensor]:
        # given a list of (query, docs), return scores
        scores = self.process_batch(sentences)
        return scores

    def predict_one(self,
                    query: str,
                    doc_list: list[str],
                    batch_size: int = None,
                    show_progress_bar: bool | None = False,
                    num_workers: int = 1,
                    activation_fct=None,
                    apply_softmax: bool | None = False,
                    convert_to_numpy: bool = True,
                    convert_to_tensor: bool = False,
                    **kwargs
                    ):
        sentences = [(query, doc_list)]
        scores = self.process_batch(sentences)
        return scores

    def stop(self):
        destroy_model_parallel()



if __name__ == '__main__':
    import argparse
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--student_path', type=str, default='...')
    base_args.add_argument('--output_training_dataset_path', type=str)
    base_args.add_argument('--save_model_path', type=str)
    args = base_args.parse_args()
    model_path = args.student_path
    dataset_path = args.output_training_dataset_path
    save_path = args.save_model_path
    wandb.init(project="reranker")
    ranker = Qwen3SetwiseRerankerInferenceModel(model_path, model_type='transformers')
    ranker.train_SFT(dataset_path=dataset_path,
                     save_path=save_path)
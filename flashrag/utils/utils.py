import os
import importlib
from transformers import AutoConfig
from flashrag.dataset.dataset import Dataset


def get_dataset(config):
    """Load dataset from config."""

    dataset_path = config["dataset_path"]
    all_split = config["split"]

    split_dict = {split: None for split in all_split}

    for split in all_split:
        split_path = os.path.join(dataset_path, f"{split}.jsonl")
        if not os.path.exists(split_path):
            print(f"{split} file not exists!")
            continue
        if split in ["test", "val", "dev"]:
            split_dict[split] = Dataset(
                config, split_path, sample_num=config["test_sample_num"], random_sample=config["random_sample"]
            )
        else:
            split_dict[split] = Dataset(config, split_path)

    return split_dict


def get_generator(config, **params):
    """Automatically select generator class based on config."""
    if config["framework"] == "vllm":
        return getattr(importlib.import_module("flashrag.generator"), "VLLMGenerator")(config, **params)
    elif config["framework"] == "fschat":
        return getattr(importlib.import_module("flashrag.generator"), "FastChatGenerator")(config, **params)
    elif config["framework"] == "hf":
        model_config = AutoConfig.from_pretrained(config["generator_model_path"])
        arch = model_config.architectures[0]
        if "t5" in arch.lower() or "bart" in arch.lower() or 'fusionindecoder' in arch.lower():
            return getattr(importlib.import_module("flashrag.generator"), "EncoderDecoderGenerator")(config, **params)
        else:
            return getattr(importlib.import_module("flashrag.generator"), "HFCausalLMGenerator")(config, **params)
    elif config["framework"] == "openai":
        return getattr(importlib.import_module("flashrag.generator"), "OpenaiGenerator")(config, **params)
    else:
        raise NotImplementedError


def get_retriever(config):
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    if config["retrieval_method"] == "bm25":
        return getattr(importlib.import_module("flashrag.retriever"), "BM25Retriever")(config)
    else:
        return getattr(importlib.import_module("flashrag.retriever"), "DenseRetriever")(config)


def get_reranker(config):
    if config["rerank_method"] == "bge":
        return getattr(importlib.import_module("flashrag.retriever"), "CrossReranker")(config)
    elif config["rerank_method"] == 'purecover':
        return getattr(importlib.import_module("flashrag.retriever"), "AttentionReranker")(config)
    else:
        return getattr(importlib.import_module("flashrag.retriever"), "BiReranker")(config)



def get_judger(config):
    judger_name = config["judger_name"]
    if "skr" in judger_name.lower():
        return getattr(importlib.import_module("flashrag.judger"), "SKRJudger")(config)
    elif "adaptive" in judger_name.lower():
        return getattr(importlib.import_module("flashrag.judger"), "AdaptiveJudger")(config)
    else:
        assert False, "No implementation!"


def get_refiner(config, retriever=None, generator=None):
    # 预定义默认路径字典
    DEFAULT_PATH_DICT = {
        "recomp_abstractive_nq": "fangyuan/nq_abstractive_compressor",
        "recomp:abstractive_tqa": "fangyuan/tqa_abstractive_compressor",
        "recomp:abstractive_hotpotqa": "fangyuan/hotpotqa_abstractive",
    }
    REFINER_MODULE = importlib.import_module("flashrag.refiner")

    refiner_name = config["refiner_name"]
    refiner_path = (
        config["refiner_model_path"]
        if config["refiner_model_path"] is not None
        else DEFAULT_PATH_DICT.get(refiner_name, None)
    )

    try:
        model_config = AutoConfig.from_pretrained(refiner_path)
        arch = model_config.architectures[0].lower()
        print(arch)
    except Exception as e:
        print("Warning", e)
        model_config, arch = "", ""

    if "recomp" in refiner_name or "bert" in arch:
        if model_config.model_type == "t5":
            refiner_class = "AbstractiveRecompRefiner"
        else:
            refiner_class = "ExtractiveRefiner"
    elif "lingua" in refiner_name:
        refiner_class = "LLMLinguaRefiner"
    elif "selective-context" in refiner_name or "sc" in refiner_name:
        refiner_class = "SelectiveContextRefiner"
    elif "kg-trace" in refiner_name:
        return getattr(REFINER_MODULE, "KGTraceRefiner")(config, retriever, generator)
    else:
        raise ValueError("No implementation!")

    return getattr(REFINER_MODULE, refiner_class)(config)


def hash_object(o) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    import hashlib
    import io
    import dill
    import base58

    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


import random
import re
import ast

def get_last_list_str(s):
    matches = re.findall(r'\[(.*?)\]', s, re.DOTALL)
    # matches = re.findall(r'(\[\s*.*?\s*\])', s, re.DOTALL)
    if matches:
        # 提取到的列表字符串
        return '['+ matches[-1] +']'
    else:
        return ""

def extract_str_to_list(list_str):
    try:
        print(f'list_str={list_str}')
        result = ast.literal_eval(list_str)
        # print(result)  # 输出: ['a', 'b']
        # print(type(result))  # 输出: <class 'list'>
        if not isinstance(result, list):
            return []
        else:
            if len(result) == 0:
                return []
            else:
                for sub_query in result:
                    if not isinstance(sub_query, str):
                        return []
        return result
    except Exception as e:
        print(f'Error Occur when extracting query:{e}')
        return []

import json
def save_json(data, filepath, indent=2, ensure_ascii=False):
    """
    将 Python 对象保存为 JSON 文件。

    参数:
        data (dict or list): 要保存的数据。
        filepath (str): 保存路径（含文件名），例如 'output/data.json'。
        indent (int): 缩进空格数，默认为 2。
        ensure_ascii (bool): 是否转义非 ASCII 字符。默认为 False，可保存中文。
    """
    # 创建父目录（如果不存在）
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    print(f"JSON saved to {filepath}")


def load_json(path, encoding='utf-8'):
    """
    加载一个 JSON 文件，返回其内容。

    参数:
        path (str): JSON 文件的路径
        encoding (str): 文件编码格式，默认 'utf-8'

    返回:
        dict/list: 解析后的 JSON 数据

    异常:
        如果文件不存在、格式错误或不是合法的 JSON，将抛出异常。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")

    with open(path, 'r', encoding=encoding) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析 JSON 失败: {e}")

    return data

def load_jsonl(filepath):
    """
    加载 .jsonl 文件，每行一个 JSON 对象，返回列表。

    Args:
        filepath (str): 文件路径

    Returns:
        List[dict]: JSON 对象组成的列表
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

import openai

def call_openai_chat(prompt, model="xxx",
                     api_key='EMPTY',
                     base_url="http://localhost:8000/v1",
                     max_tokens=None,
                     temperature=0.0):
    """
    调用 OpenAI 的 chat/completions 接口
    :param prompt: 用户输入的文本
    :param model: 使用的模型名称（如 gpt-4, gpt-3.5-turbo）
    :param api_key: 你的 OpenAI API 密钥
    :return: OpenAI 的回复字符串
    """
    client = openai.OpenAI(api_key=api_key,
                           base_url=base_url)
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    if max_tokens is not None:  # 只有在不是 None 时才加进去
        params["max_tokens"] = max_tokens
    # 调用 Chat Completions 接口
    response = client.chat.completions.create(
        **params
    )

    # 提取回复内容
    reply = response.choices[0].message.content.strip("")
    print(reply)
    # 提取模型的回复内容
    return reply

import cvxpy as cp
import numpy as np
# def max_coverage_rerank(A, w=None, K=None):
#     """
#     使用cvxpy优化最大信息覆盖问题
#     参数:
#         A: numpy.ndarray, shape (N, M), A[i, j] 表示文档i是否覆盖topic j（0/1）
#         w: numpy.ndarray, shape (M,), 每个topic的权重（可选，默认为1）
#         K: int, 最多选择的文档数（可选，不限制时为None）
#
#     返回:
#         selected_docs: 被选中的文档索引列表
#     """
#     N, M = A.shape
#     # print(f'A:{A}')
#     print(f'A.shape = {A.shape}')
#     # 定义变量
#     x = cp.Variable(N, boolean=True)  # 是否选择文档
#     y = cp.Variable(M)  # 是否覆盖 topic
#
#     # 默认所有topic权重为1
#     if w is None:
#         w = np.ones(M)
#
#     # 目标函数：最大化主题覆盖加权和
#     objective = cp.Maximize(w @ y)
#     # 约束1：如果topic被覆盖，则至少一个包含它的文档被选中
#     constraints = [
#         y[j] <= A[:, j] @ x for j in range(M)]  +[y >= 0, y <= 1,]
#     # 约束2：最多选择 K 个文档
#     if K is not None:
#         constraints.append(cp.sum(x) == K)
#
#     # 求解
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver='MOSEK')  # 或 solver='ECOS_BB' / 'SCIP' / 'CBC'
#     print(f'x.value = {x.value}')
#     selected_docs = [i for i in range(N) if x.value[i] > 0.5]
#     covered_topics = [j for j in range(M) if y.value[j] > 0.5]
#
#     return selected_docs, covered_topics


def max_coverage_rerank(A, w=None, K=None, lbd=0):
    """
    使用cvxpy优化概率最大覆盖问题（Soft Coverage）

    参数:
        A: numpy.ndarray, shape (N, M), A[i, j] ∈ [0,1]，表示文档i覆盖topic j的概率
        w: numpy.ndarray, shape (M,), 每个topic的权重（可选，默认为1）
        K: int, 最多选择的文档数（可选，不限制时为None）

    返回:
        selected_docs: 被选中的文档索引列表
        covered_scores: 每个topic被覆盖的概率（或期望权重）
    """
    N, M = A.shape
    print(f'A.shape = {A.shape}')

    # 定义变量
    x = cp.Variable(N)  # [0,1] 放松变量（是否选中文档）

    # 默认权重为1
    if w is None:
        w = np.ones(M)

    # 计算每个 topic 被覆盖的概率: y_j = 1 - Π_i (1 - A[i,j])^x_i
    y_list = []
    for j in range(M):
        log_terms = cp.sum(cp.multiply(x, np.log(1 - A[:, j] + 1e-10)))  # 避免log(0)
        y_j = 1 - cp.exp(log_terms)
        y_list.append(y_j)

    y = cp.hstack(y_list)  # shape (M,)

    # 目标函数：最大化覆盖权重
    objective = cp.Maximize(w @ y + lbd * cp.sum(A.T @ x))

    # 约束
    constraints = [
        x >= 0,
        x <= 1
    ]
    if K is not None:
        constraints.append(cp.sum(x) == K)

    # 求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='MOSEK')  # MOSEK、ECOS_BB、SCIP、CBC 都可尝试

    # print(f'x.value = {x.value}')
    selected_docs = list(np.argsort(-x.value)[:K])  # 取最大值的top-K（近似离散化）
    print(f'selected_docs = {selected_docs}')
    covered_scores = y.value

    return selected_docs, covered_scores


def max_coverage_rerank_with_varing_k(A, w=None, K=None, lbd=0):
    """
    使用cvxpy优化概率最大覆盖问题（Soft Coverage）

    参数:
        A: numpy.ndarray, shape (N, M), A[i, j] ∈ [0,1]，表示文档i覆盖topic j的概率
        w: numpy.ndarray, shape (M,), 每个topic的权重（可选，默认为1）
        K: int, 最多选择的文档数（可选，不限制时为None）

    返回:
        selected_docs: 被选中的文档索引列表
        covered_scores: 每个topic被覆盖的概率（或期望权重）
    """
    N, M = A.shape
    print(f'A.shape = {A.shape}')

    # 定义变量
    x = cp.Variable(N)  # [0,1] 放松变量（是否选中文档）

    # 默认权重为1
    if w is None:
        w = np.ones(M)

    # 计算每个 topic 被覆盖的概率: y_j = 1 - Π_i (1 - A[i,j])^x_i
    y_list = []
    for j in range(M):
        log_terms = cp.sum(cp.multiply(x, np.log(1 - A[:, j] + 1e-10)))  # 避免log(0)
        y_j = 1 - cp.exp(log_terms)
        y_list.append(y_j)

    y = cp.hstack(y_list)  # shape (M,)

    # 目标函数：最大化覆盖权重
    objective = cp.Maximize(w @ y + lbd * cp.sum(A.T @ x))

    # 约束
    constraints = [
        x >= 0,
        x <= 1
    ]
    if K is not None:
        constraints.append(cp.sum(x) <= K)

    # 求解问题
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='MOSEK')  # MOSEK、ECOS_BB、SCIP、CBC 都可尝试

    print(f'x.value = {x.value}')

    # 1. 排序 + 保留原始下标
    scores = x.value
    sorted_indices = np.argsort(-scores)  # 从大到小
    sorted_scores = scores[sorted_indices]

    # 2. 计算一阶差值（相邻差值）
    diffs = np.diff(sorted_scores)

    # 3. 找出下降幅度最大的地方（拐点）
    elbow_idx = np.argmax(np.abs(diffs)) + 1

    # 4. 截断后，返回原始的 doc 下标列表
    selected_doc_indices = sorted_indices[:elbow_idx]

    print(f"拐点位置：{elbow_idx}")
    print(f"截断后 doc 下标列表：{selected_doc_indices}")
    # covered_scores = y.value
    selected_doc_scores = scores[selected_doc_indices]
    return selected_doc_indices, selected_doc_scores #covered_scores


def submodular_greedy_soft_coverage_and_max(A, K, lmbda=1.0, threshold=1e-5, a=None):
    """
    贪心子模优化：最大化 g(S) + lambda * f(S)

    参数:
        A: np.ndarray, shape (N, M), A[i, j] 表示文档 i 覆盖 topic j 的概率分数，∈ [0,1]
        K: int, 最大选择文档数
        lmbda: float, f(S) 的权重

    返回:
        selected: list[int], 选中的文档索引
        g_val, f_val: 最终 g(S), f(S) 的值
    """
    N, M = A.shape
    if a is None:
        a = np.ones(M)
    selected = []
    remaining = set(range(N))

    # 初始化 soft coverage 概率和 max coverage 分值
    covered_prob = np.zeros(M)         # 对于 g(S)：每个 topic 的 soft coverage
    max_score_per_topic = np.zeros(M)  # 对于 f(S)：每个 topic 的最大得分
    best_gains = []
    for _ in range(K):
        best_gain = -np.inf
        best_doc = None

        for d in remaining:
            # 新 soft coverage 增量
            new_covered = np.multiply(a, 1 - (1 - covered_prob) * (1 - A[d]))  # shape (M,)
            delta_g = np.sum(new_covered - covered_prob)

            # 新最大得分 per topic 增量
            new_max = np.multiply(a, np.maximum(max_score_per_topic, A[d]))
            delta_f = np.sum(new_max - max_score_per_topic)

            gain = delta_g + lmbda * delta_f

            if gain > best_gain:
                best_gain = gain
                best_doc = d
                best_delta_g = delta_g
                best_delta_f = delta_f
                best_new_covered = new_covered
                best_new_max = new_max

        print(f'best gain = {best_gain} '
              f'best delta_g = {best_delta_g} '
              f'best delta_f = {best_delta_f} ')
        if best_doc is None or best_gain <= threshold: #1e-5
            break

        selected.append(best_doc)
        best_gains.append(best_gain)
        remaining.remove(best_doc)
        covered_prob = best_new_covered
        max_score_per_topic = best_new_max

    g_val = np.sum(covered_prob)
    f_val = np.sum(max_score_per_topic)

    return selected, best_gains, f_val



def submodular_greedy_soft_coverage_and_noise(A, K, lmbda=1.0, threshold=1e-5):
    """
    贪心子模优化：最大化 g(S) + lambda * f(S)

    参数:
        A: np.ndarray, shape (N, M), A[i, j] 表示文档 i 覆盖 topic j 的概率分数，∈ [0,1]
        K: int, 最大选择文档数
        lmbda: float, f(S) 的权重

    返回:
        selected: list[int], 选中的文档索引
        g_val, f_val: 最终 g(S), f(S) 的值
    """
    N, M = A.shape
    selected = []
    remaining = set(range(N))
    print(f'doc num = {N} cot num = {M}')
    # 初始化 soft coverage 概率和 max coverage 分值
    covered_prob = np.zeros(M)         # 对于 g(S)：每个 topic 的 soft coverage
    max_score_per_doc = 0  # 对于 f(S)：每个 topic 的最大得分
    best_gains = []
    for _ in range(K):
        best_gain = -np.inf
        best_doc = None

        for d in remaining:
            # 新 soft coverage 增量
            new_covered = 1 - (1 - covered_prob) * (1 - A[d])  # shape (M,)
            # print(f'new covered: {new_covered}')
            # print(f'covered prob: {covered_prob}')
            delta_g = np.sum(new_covered - covered_prob)
            # print(f'delta_g = {delta_g}')
            # 新最大得分 per topic 增量
            max_evidence_of_d = np.argmax(A[d])
            max_p_of_evidence = np.max(A[:, max_evidence_of_d])
            noise_d = 1 - np.max(A[d])/max_p_of_evidence
            # print(f'new max: {new_max}')
            gain = (1-lmbda) *delta_g - lmbda * noise_d

            if gain > best_gain:
                best_gain = gain
                best_doc = d
                best_delta_g = delta_g
                best_new_covered = new_covered
                best_noise_d = noise_d

        print(f'best gain = {best_gain}'
              f'best delta_g = {best_delta_g} '
              f'best noise_d = {best_noise_d} ')
        if best_doc is None: #1e-5
            break
        if best_gain <= threshold: # and len(selected) >= M
            break
        if len(selected) >= 2*M:
            break

        selected.append(best_doc)
        best_gains.append(best_gain)
        remaining.remove(best_doc)
        covered_prob = best_new_covered
    f_val = 0
    return selected, best_gains, f_val




def denoised_submodular_greedy_soft_coverage(A, noise_vector, K, lmbda=1.0, threshold=1e-5):
    """
    贪心子模优化：最大化 g(S) + lambda * f(S)
s
    参数:
        A: np.ndarray, shape (N, M), A[i, j] 表示文档 i 覆盖 topic j 的概率分数，∈ [0,1]
        noise_vector: shape (doc_num)
        K: int, 最大选择文档数
        lmbda: float, f(S) 的权重

    返回:
        selected: list[int], 选中的文档索引
        g_val, f_val: 最终 g(S), f(S) 的值
    """
    N, M = A.shape
    selected = []
    remaining = set(range(N))

    # 初始化 soft coverage 概率和 max coverage 分值
    covered_prob = np.zeros(M)         # 对于 g(S)：每个 topic 的 soft coverage
    max_score_per_doc = 0  # 对于 f(S)：每个 topic 的最大得分
    best_gains = []
    for _ in range(K):
        best_gain = -np.inf
        best_doc = None

        for d in remaining:
            # 新 soft coverage 增量
            new_covered = 1 - (1 - covered_prob) * (1 - A[d])  # shape (M,)
            # print(f'new covered: {new_covered}')
            # print(f'covered prob: {covered_prob}')
            delta_g = np.sum(new_covered - covered_prob)
            # print(f'delta_g = {delta_g}')
            # 新最大得分 per topic 增量

            noise_d = noise_vector[d]
            # print(f'new max: {new_max}')
            gain = delta_g - lmbda * noise_d

            if gain > best_gain:
                best_gain = gain
                best_doc = d
                best_delta_g = delta_g
                best_new_covered = new_covered
                best_noise_d = noise_d

        print(f'best gain = {best_gain} '
              f'best delta_g = {best_delta_g} '
              f'best noise_d = {best_noise_d} ')
        if best_doc is None or best_gain <= threshold: #1e-5
            break

        selected.append(best_doc)
        best_gains.append(best_gain)
        remaining.remove(best_doc)
        covered_prob = best_new_covered
    f_val = 0
    return selected, best_gains, f_val


def get_sub_query(query, model, base_url="http://localhost:8000/v1", api_key="EMPTY"):
    subqueries = []
    #         If the query cannot be decomposed, return a rewritten, clearer version: ["rewritten query"].
    prompt = f"""Please decompose the user query into several sub-queries.  Please decompose into as many sub-queries as possible (more than 3).
        Each sub-query is an independent question which contains information that is needed and helpful to answer the query. Please do not omit any information in any of the sub-queries.
        Return the sub-queries in a Python list format like this: ["sub-query 1", "sub-query 2", ...].

        Query: {query}
        """
    print(f'query: {query}')
    #         Here are the example sub-queries:
    #
    #         Query: What was the name of the conference in which the Vermont Catamounts men's soccer team competed from 1988 to 1996?
    #         Answer: ["Which conferences did the Vermont Catamounts men’s soccer team compete in?", "What years did the Vermont Catamounts men’s soccer team compete in each conference?"]
    #
    #         Query: Andrew Jaspan was the co-founder of what not-for-profit media outlet?
    #         Answer: ["What organizations did Andrew Jaspan co-found?", "Which of those organizations are not-for-profit media outlets?"]
    #
    #         Query: In what county is Tysons Galleria located?
    #         Answer: ["What is Tysons Galleria?", "Where is Tysons Galleria located?"]
    #
    #         Query: When was the English actress, writer, and comedian born who was a star in Girls on Top and in French and Saunders?
    #         Answer: ["Which English actress, writer, and comedian starred in Girls on Top and French and Saunders?", "When was the English actress, writer, and comedian who starred in Girls on Top and French and Saunders born?"]
    #
    count = 10
    while not subqueries and count >= 0:
        subq_text = call_openai_chat(prompt=prompt, model=model,
                                     base_url=base_url,
                                     api_key=api_key)

        last_list_str = get_last_list_str(subq_text)
        # print(f'last_list_str: {last_list_str}')
        subqueries = extract_str_to_list(last_list_str)
        count = count - 1
    #     print(f'subqueries: {subqueries}')
    # print(f'query: {query}')
    if not subqueries:
        return [query]
    print(f'subqueries: {subqueries}')
    return subqueries

import json
from typing import Union, Dict, Any, List
from pathlib import Path

def append_to_jsonl_file(
    path: Union[str, Path],
    new_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    ensure_ascii: bool = False
):
    """
    向 JSONL 文件追加条目（每行一个 JSON 对象）。
    不读取已有内容，写入速度快，适合大规模数据写入。
    """
    path = Path(path)

    if isinstance(new_data, dict):
        new_data = [new_data]

    with path.open('a', encoding='utf-8') as f:
        for record in new_data:
            json_str = json.dumps(record, ensure_ascii=ensure_ascii)
            f.write(json_str + '\n')


def split_by_steps(response_text):
    # 匹配 Step 和 Answer 两种段落
    pattern = r'(\[Step\s*\d+\]:.*?(?=\[Step\s*\d+\]:|\[Answer\]:|$)|\[Answer\]:.*?$)'
    matches = re.findall(pattern, response_text, flags=re.DOTALL)
    return [m.strip() for m in matches if m.strip()]
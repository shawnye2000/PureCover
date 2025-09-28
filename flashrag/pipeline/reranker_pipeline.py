from flashrag.pipeline.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator, get_reranker
# from flashrag.prompt import PromptTemplate

class RerankerPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)
        input_prompts = [
            self.prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, dataset.retrieval_result)
        ]


        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset
import os
from vllm import LLM, SamplingParams
from collections import defaultdict
from glob import glob
import pandas as pd
import torch
import pickle

llm = LLM(model="meta-llama/Meta-Llama-3.1-70B-Instruct", tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
tokenizer = llm.get_tokenizer()

instruction = """You are an expert evaluator tasked with assessing the quality of a model-generated answer compared to a gold standard correct answer in a long-form question-answering context. Your goal is to provide a quantified evaluation across multiple dimensions. Please follow these steps:

Carefully read the original question, the model-generated answer, and the gold correct answer. Evaluate the model-generated answer on the following dimensions, providing a score from 1-10 for each (where 1 is poor and 10 is excellent): a) Relevance (1-10): How well does the answer address the specific question asked? b) Accuracy (1-10): To what extent is the information provided correct and aligned with the gold answer? c) Completeness (1-10): How thoroughly does the answer cover all aspects of the question compared to the gold answer? d) Conciseness (1-10): Does the answer provide information efficiently without unnecessary details?

Calculate an overall quality score by taking the average of the five dimension scores. In your answer for each dimension, provide a justification why not a higher score and why not a lower score.

Question: {}

Model-generated Answer: {}

Gold Correct Answer: {}

Structure your response as follows:

Evaluation:
1. Relevance: [Score] - [Explanation]
2. Accuracy: [Score] - [Explanation]
3. Completeness: [Score] - [Explanation]
4. Conciseness: [Score] - [Explanation]

Overall Quality Score: [Average of the four above scores]"""


llm_scores_dict = {k: {} for k in ['coarsegrained']}

prompt_list = []

target_file = 'llama31_70B_llm_judge_coarsegrained.pkl'
if os.path.isfile(target_file):
    with open(target_file, 'rb') as fin:
        llm_scores_dict = pickle.load(fin)


for eval_aspect in ['coarsegrained']:
    instr = instruction
    
    for config in ["mem", "ft", "rag", "ft"]:
        print(config)
        if config in llm_scores_dict[eval_aspect]:
            pass
        else:
            llm_scores_dict[eval_aspect][config] = defaultdict(list)
        
        # Can use f"../data_resp/scidqa/{run_id}/{config}/{output_suffix}/*.xlsx" for other configurations
        files  = glob(f"../data_resp/{config}/*.xlsx")

        for file in files:
            print(file)
            prompt_list = []
            model_ans_list = []
            
            try:
                df = pd.read_excel(file)
            except Exception as e:
                print(e)
                continue
            llm_name = file.rsplit("/", 1)[1]
            llm_name = llm_name.replace(".xlsx", "")
            
            if llm_name in llm_scores_dict[eval_aspect][config]:
                print(f"{llm_name}: Already done, skipping...")
                continue
            if file.find("gpt4") > -1 or file.find("gpt-4") > -1 or file.find("gemini") > -1:
                if len([x for x in df.columns if x.find("_ans_single") > -1]) > 0:
                    llm_ans_col = [x for x in df.columns if x.find("_ans_single") > -1][0]
                elif len([x for x in df.columns if x.find("_ans") > -1]) > 0:
                    llm_ans_col = [x for x in df.columns if x.find("_ans") > -1][0]
            elif config == "ft":
                llm_ans_col = [x for x in df.columns if x.find("_ans_single") > -1][0]
            else:
                llm_ans_col = [x for x in df.columns if x.find("_ans") > -1][0]
            
            for _, row in enumerate(df.iterrows()):
                row = row[1]
                que = str(row['que'])
                try:
                    llm_resp = str(row[llm_ans_col]).strip()
                except:
                    llm_resp = "Skipped."
                gt = str(row['ans']).strip()
                if gt.startswith("A: "):
                    gt = gt[3:]
                    gt = gt.strip()
                
                local_instr = instr.format(que, llm_resp, gt)

                conversations = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': local_instr}],
                    tokenize=False,
                )
                prompt_list.append(conversations)
            
            sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=1024, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
            outputs = llm.generate(prompt_list, sampling_params)
            for output in outputs:
                model_ans_list.append(output.outputs[0].text)
            llm_scores_dict[eval_aspect][config][llm_name] = model_ans_list
            with open(target_file, 'wb') as fin:
                pickle.dump(llm_scores_dict, fin)

with open(target_file, 'wb') as fin:
    pickle.dump(llm_scores_dict, fin)
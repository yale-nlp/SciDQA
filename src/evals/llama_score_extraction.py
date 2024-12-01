import os
import pandas as pd
from glob import glob
import torch
import pickle
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
tokenizer = llm.get_tokenizer()

def get_llama31_completion(prompt_list):
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=512, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    outputs = llm.generate(prompt_list, sampling_params)
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    return responses

if __name__ == "__main__":

    prompt_list = []
    scores_dict = {"coarsegrained": {k: {} for k in ['mem', 'tabs', 'rag', 'ft']}}
    # set create_json=True to extract scores in json format
    create_json = False
    
    if create_json:
        target_file_path = "llama31_70B_llm_judge_aspect_scores_json.pkl"

        instr = """You are provided with an evaluation of an answer in the following format:
    
        Evaluation:
        1. Relevance: [Score] - [Explanation]
        2. Accuracy: [Score] - [Explanation]
        3. Completeness: [Score] - [Explanation]
        4. Conciseness: [Score] - [Explanation]
        Overall Quality Score: [Average of the four above scores].

        Carefully read the evaluation provided next, and extract the scores for each aspect in a json format as follows:
        {'Relevance': [Score], 'Accuracy': [Score], 'Completeness': [Score], 'Conciseness': [Score], 'Overall Quality Score': [Score]}

        Only extract the scores to create the json and do not include any explanation.
        """
    else:
        target_file_path = "llama31_70B_llm_judge_aspect_scores.pkl"
        instr = """You are provided with an evaluation of an answer in the following format:
    
        Evaluation:
        1. Relevance: [Score] - [Explanation]
        2. Accuracy: [Score] - [Explanation]
        3. Completeness: [Score] - [Explanation]
        4. Conciseness: [Score] - [Explanation]
        Overall Quality Score: [Average of the four above scores].

        Carefully read the evaluation provided next, and extract the final overall quality score from the discussion. Do not include any explanation, you should only provide the final numeric score for overall quality from the evaluation statement.
        """
    
    if os.path.exists(target_file_path):
        with open(target_file_path, "rb") as fout:
            scores_dict = pickle.load(fout)

    
    evaluation_file_path = "llama31_70B_llm_judge_coarsegrained.pkl"
    with open(evaluation_file_path, 'rb') as fin:
        llmjudge_scores = pickle.load(fin) 

    for eval_aspect in ['coarsegrained']:
        for config in ['mem', 'tabs', 'rag', 'ft']:
            for llm_name in llmjudge_scores[eval_aspect][config]:
                if llm_name in scores_dict[eval_aspect][config]: 
                    print(llm_name, " - Already done, skipping...")
                    continue
                print(llm_name, " - Extracting overall scores...")
                prompt_list = []
                for _, eval_statement in enumerate(llmjudge_scores[eval_aspect][config][llm_name]):
                    prompt = instr + "\n\n" + eval_statement
                    conversations = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}],
                    tokenize=False,
                    )
                    prompt_list.append(conversations)

                completion = get_llama31_completion(prompt_list)
                scores_dict[eval_aspect][config][llm_name] = completion
                with open(target_file_path, "wb") as fout:
                    pickle.dump(scores_dict, fout)
                
    with open(target_file_path, "wb") as fout:
        pickle.dump(scores_dict, fout)
    
    print("Scores extracted and saved successfully!")
    print(target_file_path)
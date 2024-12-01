import os
os.environ['CURL_CA_BUNDLE'] = ''
from vllm import LLM, SamplingParams
import xlsxwriter
import argparse
import time
import torch
import pickle
import pandas as pd
from collections import defaultdict
import json
import re

# Function to remove non-printable characters
def remove_illegal_characters(text):
    if isinstance(text, str):
        # Define what characters are considered illegal (here we remove all non-printable characters)
        return re.sub(r'[^\x20-\x7E]', '', text)  # Removes non-ASCII printable characters
    return text


model_name_map = {"meta-llama/Llama-2-7b-chat-hf": "llama2_7b_chat", "meta-llama/Llama-2-13b-chat-hf": "llama2_13b_chat", "meta-llama/Llama-2-70b-chat-hf": "llama2_70b_chat", "meta-llama/Llama-2-7b-hf": "llama2_7b", "meta-llama/Llama-2-13b-hf": "llama2_13b", "meta-llama/Llama-2-70b-hf": "llama2_70b", "mistralai/Mistral-7B-v0.1": "mistral_7b", "mistralai/Mistral-7B-Instruct-v0.1": "mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.2": "mistral_7b_instruct_v2", "mistralai/Mixtral-8x7B-v0.1": "mistral_8_7b", "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistral_8_7b_instruct", "lmsys/vicuna-13b-v1.5": "vicuna_13b", "lmsys/vicuna-13b-v1.5-16k": "vicuna_13b_16k", "lmsys/longchat-7b-v1.5-32k": "longchat_7b_32k", "lmsys/vicuna-7b-v1.5": "vicuna_7b", "lmsys/vicuna-7b-v1.5-16k": "vicuna_7b_16k", "HuggingFaceH4/zephyr-7b-beta": "zephyr_7b_beta", "tiiuae/falcon-7b": "falcon_7b", "tiiuae/falcon-7b-instruct": "falcon_7b_instruct", "tiiuae/falcon-40b": "falcon_40b", "facebook/galactica-6.7b": "galactica_7b", "facebook/galactica-30b": "galactica_30b", "microsoft/phi-2": "ms_phi2_3b", "google/gemma-2b-it": "gemma_2b_it", "google/gemma-2b": "gemma_2b", "Qwen/Qwen2-beta-7B-Chat": "qwen_7b_chat", "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b", "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama31_8b", "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama31_70b",
"Qwen/Qwen2.5-1.5B-Instruct": "qwen_1pt5B_IT", "Qwen/Qwen2.5-7B-Instruct": "qwen_7B_IT"}

model2contextlength = {
    "Qwen/Qwen2-beta-7B-Chat": ("qwen_7b_chat", 8192, "Qwen/Qwen2-beta-7B-Chat"),
    "meta-llama/Llama-2-7b-chat-hf": ("llama2_7b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-13b-chat-hf": ("llama2_13b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/blob/main/generation_config.json"), 
    "meta-llama/Llama-2-70b-chat-hf": ("llama2_70b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-7b-hf": ("llama2_7b", 4096, "https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-13b-hf": ("llama2_13b", 4096, "https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-70b-hf": ("llama2_70b", 4096, "https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/generation_config.json"),
    "mistralai/Mistral-7B-v0.1": ("mistral_7b", 8000, "https://aws.amazon.com/blogs/machine-learning/mistral-7b-foundation-models-from-mistral-ai-are-now-available-in-amazon-sagemaker-jumpstart/#:~:text=Mistral%207B%20has%20an%208%2C000,at%20a%207B%20model%20size."),
    "mistralai/Mistral-7B-Instruct-v0.1": ("mistral_7b_instruct", 8000, "https://www.secondstate.io/articles/mistral-7b-instruct-v0.1/"),
    "mistralai/Mixtral-8x7B-v0.1": ("mistral_8_7b", 32000, "https://mistral.ai/news/mixtral-of-experts/"), 
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ("mistral_8_7b_instruct", 32000, "https://mistral.ai/news/mixtral-of-experts/"),
    "lmsys/vicuna-13b-v1.5": ("vicuna_13b", 4096, "https://huggingface.co/lmsys/vicuna-13b-v1.5/blob/main/generation_config.json"),
    "lmsys/vicuna-13b-v1.5-16k": ("vicuna_13b_16k", 16384, "https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/blob/main/generation_config.json"), 
    "lmsys/longchat-7b-v1.5-32k": ("longchat_7b_32k", 32768, "https://huggingface.co/lmsys/longchat-7b-v1.5-32k/blob/main/tokenizer_config.json"),
    "lmsys/vicuna-7b-v1.5": ("vicuna_7b", 4096, "https://huggingface.co/lmsys/vicuna-7b-v1.5/blob/main/generation_config.json"),
    "lmsys/vicuna-7b-v1.5-16k": ("vicuna_7b_16k", 16384, "https://huggingface.co/lmsys/vicuna-7b-v1.5-16k/blob/main/generation_config.json"), 
    "HuggingFaceH4/zephyr-7b-beta": ("zephyr_7b_beta", 16384, "https://docs.endpoints.anyscale.com/supported-models/huggingfaceh4-zephyr-7b-beta/"),
    "tiiuae/falcon-7b": ("falcon_7b", 2048, "https://huggingface.co/tiiuae/falcon-7b/blob/main/tokenizer_config.json"),
    "tiiuae/falcon-7b-instruct": ("falcon_7b_instruct", 2048, "https://huggingface.co/tiiuae/falcon-7b-instruct/blob/main/tokenizer_config.json"), 
    "facebook/galactica-6.7b": ("galactica_7b", 2048, "https://llm.extractum.io/model/facebook%2Fgalactica-6.7b,11ptgQY4r8q8sc7KY9iN38"),
    "facebook/galactica-30b": ("galactica_30b", 2048, "https://llm.extractum.io/model/facebook%2Fgalactica-6.7b,11ptgQY4r8q8sc7KY9iN38"),
    "microsoft/phi-2": ("ms_phi2_3b", 2048, "https://huggingface.co/microsoft/phi-2"),
    "tiiuae/falcon-40b": ("falcon_40b", 2048, "https://huggingface.co/tiiuae/falcon-40b/blob/main/tokenizer_config.json"),
    "google/gemma-2b-it": ("gemma_2b_it", 8192, ""), 
    "google/gemma-2b": ("gemma_2b", 8192, ""), 
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ("llama31_8b", 4096, ""),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ("llama31_70b", 4096, ""),
    "Qwen/Qwen2.5-1.5B-Instruct": ("qwen_1pt5B_IT", 128000, "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"), 
    "Qwen/Qwen2.5-7B-Instruct": ("qwen_7B_IT", 128000, "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct")
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=list(model_name_map.keys()), help='name of the model to use for generation, specifically the hf repo name')
    parser.add_argument('--data_file', help='path of the dataset file')
    parser.add_argument('--resp_dir', help='path of the directory to store responses')
    parser.add_argument('--run_id', help='run id (used for response dir naming)')
    parser.add_argument('--config', choices=['mem', 'tabs', 'ft', 'rag'], default='mem', help='what information to include in the prompt')
    parser.add_argument('--alt_manuscript_version', choices=['True', 'False'], default=False)
    args = parser.parse_args()
    return args

def get_tabs(pid):
    with open('../data/relevant_ptabs.pkl', 'rb') as fin:
        ptabs = pickle.load(fin)
    title = ptabs[pid]['title']
    abstract = ptabs[pid]['abs']
    return title, abstract

if os.path.exists("../data/section_chunk_tokens_ss.pkl"):
    with open("../data/section_chunk_tokens_ss.pkl", "rb") as fin:
        tokens_data = pickle.load(fin)
else:
    print("Please run the script to generate the section chunk tokens first.")
    tokens_data = {}

if os.path.exists("../data/model_len_ft_chunks.pkl"):
    with open("../data/model_len_ft_chunks.pkl", "rb") as fin:
        chunks_data = pickle.load(fin)
else:
    print("Please run the script to generate the section chunk tokens first.")
    chunks_data = {}

if os.path.exists("../data/model_len_rag_chunks.pkl"):
    with open("../data/model_len_rag_chunks.pkl", "rb") as fin:
        rag_chunks_data = pickle.load(fin)
else:
    print("Please run the script to generate the section chunk tokens first.")
    rag_chunks_data = {}

# This loads the precomputed length adjusted chunks on the fly instead of loading different sections and then arranging them max_model_length wise. The chunk_data dict stores for each model and paper version, the length adjust sections.
def get_chunks(model_name, pid, version, alt_manuscript_version=False):
    global chunks_data
    version = version.lower().strip()
    if version == "revised":
        version = "final"
    
    # Swap manuscript version
    if alt_manuscript_version:
        if version == "initial":
            version = "final"
        else:
            version = "initial"
    return chunks_data[model_name][version][pid]

def get_rag_chunks(model_name, pid, version, id):
    global rag_chunks_data
    version = version.lower().strip()
    if version == "revised":
        version = "final"
    return rag_chunks_data[model_name][f"{pid}_{version}_{id}"]

def generate_responses(model_name, data_file, resp_dir, run_id, config, alt_manuscript_version):
    alt_manuscript_version = bool(alt_manuscript_version)
    if data_file.find("all_versions") > -1:
        # The all_versions excel sheet contains the original as well as edited data.
        question_type = "org_que"
    else:
        question_type = "que"
    
    qa_df = pd.read_excel(data_file)
    dataset_version = data_file.rsplit("/", 1)[-1]
    dataset_version = dataset_version.replace(".xlsx", "")
    response_file_name = model_name_map[model_name]
    num_cuda = torch.cuda.device_count()

    output_suffix = "prompt_beg" # other options are: single_ans, prompt_end
    response_file_name += f"_{config}"

    llm_context = ""

    if model_name in ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b"]: 
        llm = LLM(model=model_name, tensor_parallel_size=1, dtype="half")
    elif model_name in ["microsoft/phi-2"]:
        llm = LLM("microsoft/phi-2",  gpu_memory_utilization=0.95, tensor_parallel_size=num_cuda, trust_remote_code=True)
    elif model_name in ["mistralai/Mistral-7B-v0.1"]:
        llm = LLM(model=model_name, tokenizer_mode="mistral")
    elif model_name in ["mistralai/Mistral-7B-Instruct-v0.1"]:
        llm = LLM(model=model_name, tokenizer_mode="mistral")
    elif model_name in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:
        llm = LLM(model=model_name, tensor_parallel_size=2, trust_remote_code=True)
    elif llm_context == "KV cache issue" or model_name in ["meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        llm = LLM(model=model_name, max_model_len=4096, gpu_memory_utilization=0.95, max_num_batched_tokens=4096, tensor_parallel_size=num_cuda, trust_remote_code=True)
    else:
        llm = LLM(model=model_name, gpu_memory_utilization=0.95, tensor_parallel_size=num_cuda, trust_remote_code=True)
    
    print(model_name)
    model_ans_list = []
    prompts_list = []
    qid_list = []
    pid_num_sections_list = []
    if run_id == "0":
        temp = 0.1
        topp = 0.2
    elif run_id == "1":
        temp = 0.1
        topp = 0.3
    elif run_id == "2":
        temp = 0.1
        topp = 0.9
    for _, row in enumerate(qa_df.iterrows()):
        row = row[1]
        if config == "mem":
            prompt_text = f"You are provided with a question about a research paper submitted to a top-tier computer science conference. Your task is to answer the question based on your extensive knowledge of machine learning and deep learning. Question: {str(row[question_type])}" 
            prompts_list.append(prompt_text)
        elif config == "tabs":
            title, abstract = get_tabs(row['pid'])
            prompt_text = f"You are provided with the title and abstract of a research paper submitted to a top-tier computer science conference, alongwith a question. Your task is to answer the question based on your extensive knowledge of machine learning and deep learning.\nTitle: {title}\nAbstract: {abstract}\nQuestion: {str(row[question_type])}"
            prompts_list.append(prompt_text)
        elif config == "ft":
            paper_sections = get_chunks(model_name, row['pid'], row['version'])
            pid_num_sections_list.append(len(paper_sections))
            for sec in paper_sections:
                prompt_text = f"You are provided with an excerpt from a research paper submitted to a top-tier computer science conference in the domain of ML and DL. You are also provided a question. Your task is to answer the question based on your knowledge and the text provided. Do not include any additional text other than the answer. \nExcerpt: {sec}\nQuestion: {str(row[question_type])}\nAnswer: "
                prompts_list.append(prompt_text)
        elif config == "rag":
            paper_section = get_rag_chunks(model_name, row['pid'], row['version'], row['id'])[0].strip()
            prompt_text = f"You are provided with an excerpt from a research paper submitted to a top-tier computer science conference in the domain of ML and DL. You are also provided a question. Your task is to answer the question based on your knowledge and the text provided. Do not include any additional text other than the answer.\nExcerpt: {paper_section}\nQuestion: {str(row[question_type])}\nAnswer: "
            prompts_list.append(prompt_text)
            
        version_name = f"{dataset_version}/{run_id}"
    
    sampling_params = SamplingParams(temperature=temp, top_p=topp, max_tokens=512)
    outputs = llm.generate(prompts_list, sampling_params)
    if config == "ft": 
        for num_sects in pid_num_sections_list:
            same_que_ans = []
            for i in range(0, num_sects):
                same_que_ans.append(outputs.pop(0).outputs[0].text)
            model_ans_list.append(same_que_ans)
    else:
        for output in outputs:
            model_ans_list.append(output.outputs[0].text)
    
    # Save the file at the appropriate configuration. Version is dataversion_runid, config is mem/tabs/rag/ft, output_suffix is prompt_beg/alt_manuscript_version/single_ans/prompt_end
    with pd.ExcelWriter(f'{resp_dir}/{version_name}/{config}/{output_suffix}/{response_file_name}.xlsx') as writer:
        qa_df[f"{model_name_map[model_name]}_ans"] = model_ans_list
        try:
            qa_df.to_excel(writer, sheet_name=model_name_map[model_name])
        except:
            try:
                qa_df.to_excel(writer, sheet_name=model_name_map[model_name], engine='xlsxwriter')
            except:
                print("Illegal chars in answers for ", model_name)
                try:
                    qa_df = qa_df.applymap(remove_illegal_characters)
                    qa_df.to_excel(writer, sheet_name=model_name_map[model_name])
                except:
                    pass
    qa_df.to_pickle(f'{resp_dir}/{version_name}/{config}/{output_suffix}/{response_file_name}.pkl')
    return

if __name__ == "__main__":
    args = parse_args()
    generate_responses(args.model_name, args.data_file, args.resp_dir, args.run_id, args.config, args.alt_manuscript_version)

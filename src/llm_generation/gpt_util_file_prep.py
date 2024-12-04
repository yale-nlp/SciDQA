import os
import pandas as pd
import numpy as np
import pickle
import json
from openai import AzureOpenAI
from collections import Counter
import pathlib 

def get_tabs(pid):
    with open('../data/relevant_ptabs.pkl', 'rb') as fin:
        ptabs = pickle.load(fin)
    title = ptabs[pid]['title']
    abstract = ptabs[pid]['abs']
    return title, abstract

def save_to_multiple_files(prompts_list, config, AZ_MODEL):
    # Maximum file size in bytes (24MB)
    MAX_FILE_SIZE = 24 * 1024 * 1024

    pathlib.Path('./data/ft').mkdir(parents=True, exist_ok=True)
    # Base filename for output files
    base_filename = f"data/ft/{config}_{AZ_MODEL}"
    
    # Counter for file naming
    file_counter = 1

    # Open the first file
    current_file = open(f"{base_filename}_{file_counter}.jsonl", "w")
    current_file_size = 0

    # Iterate over the list of JSON objects
    for json_obj in prompts_list:
        # Convert the JSON object to a JSON string
        json_str = json.dumps(json_obj) + "\n"
        
        # Check the size of the string
        json_str_size = len(json_str.encode('utf-8'))  # Get size in bytes

        # If adding this object would exceed the max file size, open a new file
        if current_file_size + json_str_size > MAX_FILE_SIZE:
            current_file.close()  # Close the current file
            file_counter += 1  # Increment the file counter
            current_file = open(f"{base_filename}_{file_counter}.jsonl", "w")  # Open a new file
            current_file_size = 0  # Reset the size counter

        # Write the JSON string to the file
        current_file.write(json_str)
        
        # Update the current file size
        current_file_size += json_str_size

    # Close the last file
    current_file.close()

qa_df = pd.read_excel("../data/scidqa.xlsx")

with open("../data/model_len_rag_chunks.pkl", "rb") as fin:
    rag_chunks_data = pickle.load(fin)
def get_rag_chunks(model_name, pid, version, id):
    global rag_chunks_data
    version = version.lower().strip()
    if version == "revised":
        version = "final"
    return rag_chunks_data[model_name][f"{pid}_{version}_{id}"]

with open("../data/model_len_ft_chunks.pkl", "rb") as fin:
    chunks_data = pickle.load(fin)
def get_chunks(model_name, pid, version, alt_manuscript_version=False):
    global chunks_data
    version = version.lower().strip()
    if version == "revised":
        version = "final"
    # Swap manuscript version - for config where we use the other version to generate the answer
    if alt_manuscript_version:
        if version == "initial":
            version = "final"
        else:
            version = "initial"
    return chunks_data[model_name][version][pid]

exp_configs = ["mem", "tabs", "rag", "ft"]
AZ_MODEL = "gpt-4o" # other option is to use gpt-4o-mini-2

for config in exp_configs:
    prompts_list = []
    question_type = "que"
    for _, row in enumerate(qa_df.iterrows()):
        row = row[1]
        qid = config + "_" + str(_)
        if config == "mem":
            instr_prompt_text = f"You are provided with a question about a research paper submitted to a top-tier computer science conference. Your task is to answer the question based on your extensive knowledge of machine learning and deep learning."
            body_text =  f"Question: {str(row[question_type])}" 
            
        elif config == "tabs":
            title, abstract = get_tabs(row['pid'])
            instr_prompt_text = f"You are provided with the title and abstract of a research paper submitted to a top-tier computer science conference, alongwith a question. Your task is to answer the question based on your extensive knowledge of machine learning and deep learning."
            body_text = f"Title: {title}\nAbstract: {abstract}\nQuestion: {str(row[question_type])}"
        
        elif config == "rag":
            paper_section = get_rag_chunks("GPT4omini", row['pid'], row['version'], row['id'])[0].strip()
            instr_prompt_text = f"You are provided with an excerpt from a research paper submitted to a top-tier computer science conference in the domain of ML and DL. You are also provided a question. Your task is to answer the question based on your knowledge and the text provided. Do not include any additional text other than the answer."
            body_text = f"Excerpt: {paper_section}\nQuestion: {str(row[question_type])}"
        
        elif config == "ft":
            # We have the full-text saved for gemini model, and that can be directly used by any model that uses full-text.
            paper_ft = get_chunks('Gemini', row['pid'], row['version'])[0]
            instr_prompt_text = f"You are provided with text of a research paper submitted to a top-tier computer science conference in the domain of ML and DL. You are also provided a question. Your task is to answer the question based on your knowledge and the text provided. Do not include any additional text other than the answer."
            body_text = f"Excerpt: {paper_ft}\n\nQuestion: {str(row[question_type])}"
            
        
        local_prompt = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": AZ_MODEL, "temperature": 0.1, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": ""}],"max_tokens": 512}}
        local_prompt["custom_id"] = qid
        local_prompt["body"]["messages"][0]["content"] += instr_prompt_text
        local_prompt["body"]["messages"][1]["content"] = body_text
        prompts_list.append(local_prompt)

    pathlib.Path('./data/').mkdir(parents=True, exist_ok=True)
    # Save the batch file (into muliple 24MB files if FT config, otherwise a single file)
    if config == "ft":
       save_to_multiple_files(prompts_list, config, AZ_MODEL)
    else:
        file_name = f"./data/{config}_{AZ_MODEL}.jsonl"
        with open(file_name, 'w') as file:
            for obj in prompts_list:
                file.write(json.dumps(obj) + '\n')
import sys
import os
import pandas as pd 
import nltk
import pickle
import re
from collections import defaultdict
from transformers import AutoTokenizer
from model_lists import model_name_map, model2contextlength

access_token = "" 
if access_token == "":
    print("Please set huggingface access token ...")
    sys.exit(1)

inst_size = 500

with open("../data/papers_fulltext_nougat.pkl", "rb") as fin:
    papers_fulltext_dict = pickle.load(fin)

qa_df = pd.read_excel("../data/scidqa.xlsx")

def full_text_cleanup(paper_fulltext, pid, version):
    if pid in ["rJlcV2Actm", "B1x1ma4tDr"] and version == "initial":
        return [paper_fulltext]
    noisy_strinfs = ["\(\boldsymbol{\widehat{\texttt{{}}}}\)\(\boldsymbol{\widehat{\texttt{{}}}}\)\(\boldsymbol{\widehat{\texttt{{}}}}\)", "{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]", "\right.\left.\begin{array}{ccc}-1&-1\end{array} \right.\left.\begin{array}{ccc}-1&-1\end{array}", "subsubsec:subsubsec:subsec:", "[\![\![\!", "**-**: **-**: ", "eq: eq: ", "eq:eq:",
    "subsubsec:subsec:subsec:", "\normalsize\normalsize\normalsize", "Peq:Peq:Peq:", "eq_2nd_eq_2nd_eq_2nd", "lem:lem:lem:", "SS2.1, SS2.1, SS2.1, ", "{1}{1}{1}{1}{1}"]
    for un_st in noisy_strinfs:
        if paper_fulltext.find(un_st) > -1:
            paper_fulltext = paper_fulltext.replace(un_st, "")
    res = re.findall(r"\[[0-9, ]+ [0-9]+\]?", paper_fulltext)
    if res:
        for result in res:
            str_to_replace = result
            if result.startswith("["):
                str_to_replace = str_to_replace[1:]
            if result.endswith("]"):
                str_to_replace = str_to_replace[:-1]
            num_refs = str_to_replace.split(", ")
            # Remove hallucinated references
            if len(num_refs) > 15:
                num_refs = num_refs[0:5]
                paper_fulltext = paper_fulltext.replace(str_to_replace,  "["+", ".join(num_refs)+"]")
    return [paper_fulltext]

def get_ft(pid, version):
    global papers_fulltext_dict
    if pid in papers_fulltext_dict[version]:
        paper_fulltext = papers_fulltext_dict[version][pid]
        paper_fulltext = paper_fulltext.replace("\n\n\n", "\n\n")
        paper_fulltext = paper_fulltext.replace("\n\n\n", "\n\n")
        return paper_fulltext
    else:
        return ""

with open("./data/section_chunk_tokens_ss.pkl", "rb") as fin:
    tokens_data = pickle.load(fin)

def create_chunks():
    tokens_data = {mod: {"final": defaultdict(list), "initial": defaultdict(list)} for mod in model_name_map}
    if os.path.exists("../data/section_chunk_tokens.pkl"):
        with open("../data/section_chunk_tokens.pkl", "rb") as fin:
            tokens_data = pickle.load(fin)

    for mname in model_name_map:
        tokens_data[mname] = {"final": defaultdict(list), "initial": defaultdict(list)}
        if mname == "Gemini":
            # Gemini fits entire paper in context. GPT is handled separately in GPT data generation scripts. And for other models like Llama, we still need to chunk.
            for pid in qa_df['pid'].unique():
                for version in ["final", "initial"]:
                    paper_ft = get_ft(pid, version)
                    tokens_data[mname][version][pid] = full_text_cleanup(paper_ft, pid, version)        
            continue
        tokenizer = AutoTokenizer.from_pretrained(mname)
        print(f"Loaded tokenizer: {mname}")
        for pid in qa_df['pid'].unique():
            for version in ["final", "initial"]:
                paper_ft = papers_fulltext_dict[version][pid]
                paper_chunks_list = paper_ft.split("\n\n")
                for paper_chunk in paper_chunks_list:
                    chunk_toks_size = len(tokenizer.tokenize(paper_chunk))
                    if chunk_toks_size >= (model2contextlength[mname][1]-inst_size):
                        smaller_chunks = paper_chunk.split("\n")
                        for sc in smaller_chunks:
                            cs_tok_size = len(tokenizer.tokenize(sc))
                            if cs_tok_size >= (model2contextlength[mname][1]-inst_size):
                                if pid in ["rJlcV2Actm", "B1x1ma4tDr"] and version == "initial":
                                    pass
                                else:
                                    noisy_strinfs = ["\(\boldsymbol{\widehat{\texttt{{}}}}\)\(\boldsymbol{\widehat{\texttt{{}}}}\)\(\boldsymbol{\widehat{\texttt{{}}}}\)", "{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]{R_T+1}+\includegraphics[width=14.226378pt]{R_T+1}+ \includegraphics[width=14.226378pt]", "\right.\left.\begin{array}{ccc}-1&-1\end{array} \right.\left.\begin{array}{ccc}-1&-1\end{array}", "subsubsec:subsubsec:subsec:", "[\![\![\!", "**-**: **-**: ", "eq: eq: ", "eq:eq:",
                                    "subsubsec:subsec:subsec:", "\normalsize\normalsize\normalsize", "Peq:Peq:Peq:", "eq_2nd_eq_2nd_eq_2nd", "lem:lem:lem:", "SS2.1, SS2.1, SS2.1, ", "{1}{1}{1}{1}{1}"]
                                    for un_st in noisy_strinfs:
                                        if sc.find(un_st) > -1:
                                            sc = sc.replace(un_st, "")
                                    res = re.findall(r"\[[0-9, ]+ [0-9]+\]?", sc)
                                    if res:
                                        for result in res:
                                            str_to_replace = result
                                            if result.startswith("["):
                                                str_to_replace = str_to_replace[1:]
                                            if result.endswith("]"):
                                                str_to_replace = str_to_replace[:-1]
                                            num_refs = str_to_replace.split(", ")
                                            # Hallucinated references present, crop them
                                            if len(num_refs) > 15:
                                                num_refs = num_refs[0:5]
                                                sc = sc.replace(str_to_replace,  "["+", ".join(num_refs)+"]")
                                    sc_len = len(tokenizer.tokenize(sc))
                                    if sc_len < (model2contextlength[mname][1]-inst_size):
                                        tokens_data[mname][version][pid].append((sc, sc_len))
                                    else:
                                        newline_splits = sc.split(" \\\\ ")
                                        for ns in newline_splits:
                                            if len(tokenizer.tokenize(ns)) < (model2contextlength[mname][1]-inst_size):
                                                tokens_data[mname][version][pid].append((ns, len(tokenizer.tokenize(ns))))
                                            else:
                                                sent_list = nltk.sent_tokenize(ns)
                                                for sent in sent_list:
                                                    sent_len = len(tokenizer.tokenize(sent))
                                                    if sent_len < (model2contextlength[mname][1]-inst_size):
                                                        tokens_data[mname][version][pid].append((sent, sent_len))
                                                    else:
                                                        pass
                            else:
                                tokens_data[mname][version][pid].append((sc, cs_tok_size))
                    else:
                        tokens_data[mname][version][pid].append((paper_chunk, chunk_toks_size))
        with open("../data/section_chunk_tokens.pkl", "wb") as fin:
            pickle.dump(tokens_data, fin)

    with open("../data/section_chunk_tokens.pkl", "wb") as fin:
        pickle.dump(tokens_data, fin)
    
    return

def merge_tokens(tokens_data, model_name, pid, version):
    version = version.lower().strip()
    if version == "revised":
        version = "final"
    mod_len_chunks = []

    paper_chunks = tokens_data[model_name][version][pid]
    if model_name == "Gemini":
        # In case of gemini, there is only one chunk of entire full-text
        mod_len_chunks = paper_chunks
        return mod_len_chunks
    max_tokens = model2contextlength[model_name][1]
    local_chunk = ""
    len_local_chunk = 0
    for _, pc in enumerate(paper_chunks):
        # print(_)
        if (500 + len_local_chunk + pc[1]) < max_tokens:
            local_chunk = local_chunk + " " + pc[0]
            len_local_chunk += pc[1]
        else:
            if local_chunk:
                mod_len_chunks.append(local_chunk)
                local_chunk = pc[0]
                len_local_chunk = pc[1]
            else:
                print(f"Check for {model_name}, {pid}, {version}")
                pass
    return mod_len_chunks

def resize_fulltext_chunks():
    if os.path.isfile("../data/model_len_ft_chunks.pkl"):
        with open("../data/model_len_ft_chunks.pkl", "rb") as f:
            all_model_ftsecs = pickle.load(f)
    else:
        all_model_ftsecs = {modelname: {"final": defaultdict(list), "initial": defaultdict(list)} for modelname in model2contextlength}
    
    for model_name in model2contextlength:
        if model_name in all_model_ftsecs and model_name!= "Gemini":
            print("Already done: ", model_name)
            continue
        
        all_model_ftsecs[model_name] = {"final": defaultdict(list), "initial": defaultdict(list)}
        for _, row in enumerate(qa_df.iterrows()):
            row = row[1]
            version = row['version'].lower().strip()
            if version == "revised":
                version = "final"
            pid = row["pid"]
            paper_sections = (model_name, row['pid'], row['version'])
            all_model_ftsecs[model_name][version][pid] = paper_sections
    with open("../data/model_len_ft_chunks.pkl", "wb") as f:
        pickle.dump(all_model_ftsecs, f)

if __name__ == "__main__":
    create_chunks()
    resize_fulltext_chunks(tokens_data)
    
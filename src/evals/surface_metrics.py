import os
import time
from glob import glob

import numpy as np
import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer

rscorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bertscore = None
bleurt = None

def calc_rlscore(gen_ans, ans):
    try:
        ans = str(ans).strip()
        gen_ans = str(gen_ans).strip()
        scores = rscorer.score(ans, gen_ans)
        return scores['rouge1'].precision*100, scores['rouge2'].precision*100, scores['rougeL'].precision*100
    except Exception as ex:
        print("ROUGE Error: ", ex)
        print(gen_ans, ans)
        return 0.0, 0.0, 0.0

def calc_bertscore(predictions, references):
    try:
        results = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli", batch_size=1) #lang="en-sci")
        return results['precision'][0]*100, results['recall'][0]*100, results['f1'][0]*100
    except Exception as ex:
        print("BERTSCore Error: ", ex)
        print(predictions, references)
        return 0.0, 0.0, 0.0

def calc_bleurtscore(predictions, references):
    try:
        results = bleurt.compute(predictions=predictions, references=references)
        return max(results["scores"])*100
    except Exception as ex:
        print("calc_bleurtscore Error: ", ex)
        print(predictions, references)
        return 0.0

def run_evals(keep_old_scores=True):
    global rscorer, bleurt, bertscore

    all_metrics = ["r1", "r2", "rl", "bleurt", "bs_precision", "bs_recall", "bs_f1"]
    skip_metrics = [] # If the metric is already computed, skip it by adding it to this list.
    
    for config in ["mem", "tabs", "rag", "ft"]:
        fpath = f"../data_resp/{config}/*.xlsx"
        model_response_files = glob(fpath)
        model_gen_ans_col = "_ans" # all columns with the model-generated response end with this identifier.

        # Read the first dataframe and then append other model ans columns to the same df.
        cons_df = pd.read_excel(model_response_files[0])
        cons_df = cons_df.sort_values('id')

        all_model_names = []
        for i in list(cons_df.columns):
            if i.endswith(model_gen_ans_col):
                all_model_names.append(i)
        
        # append other model ans columns to the same df.
        for resp_file in model_response_files[1:]:
            try:
                resp_df = pd.read_excel(resp_file)
                resp_df = resp_df.sort_values('id')
                for df_col in list(resp_df.columns):
                    if df_col.endswith(model_gen_ans_col):
                        model_ans_col = df_col
                        all_model_names.append(df_col)
                        break
                cons_df[model_ans_col] = resp_df[model_ans_col]
            except Exception as ex:
                print(resp_file, ex)
        
        # Load existing computed scores df if it exists.
        if os.path.isfile(f"../results/{config}_eval_scores.xlsx"):
            org_df = pd.read_excel(f"../results/{config}_eval_scores.xlsx")
            org_df = org_df.sort_values('id')
        if keep_old_scores and os.path.isfile(f"../results/{config}_eval_scores.xlsx"):
            for mname in all_model_names:
                for metric in all_metrics:
                    if f"{mname}_{metric}" in org_df:
                        skip_metrics.append(f"{mname}_{metric}")
        
        if os.path.isfile(f"../results/{config}_eval_scores.xlsx"):
            org_df = pd.read_excel(f"../results/{config}_eval_scores.xlsx")
            for coli in cons_df.columns:
                if coli in org_df.columns:
                    pass
                else:
                    org_df[coli] = cons_df[coli]
            org_df.to_excel(f"../results/{config}_eval_scores.xlsx", index=False)
            cons_df = org_df.copy()
        else:
            cons_df.to_excel(f"../results/{config}_eval_scores.xlsx", index=False)

        print("Calculating rouge...")
        for mname in all_model_names:
            print(mname)
            if f"{mname}_r1" in skip_metrics and f"{mname}_r2" in skip_metrics and f"{mname}_rl" in skip_metrics:
                cons_df[f"{mname}_r1"] = org_df[f"{mname}_r1"]
                cons_df[f"{mname}_r2"] = org_df[f"{mname}_r2"]
                cons_df[f"{mname}_rl"] = org_df[f"{mname}_rl"]
                continue
            dd = cons_df.apply(lambda x: calc_rlscore(x[mname], x.ans), axis=1)
            cons_df[f"{mname}_r1"] = list(np.array([*dd.values])[:,0])
            cons_df[f"{mname}_r2"] = list(np.array([*dd.values])[:,1])
            cons_df[f"{mname}_rl"] = list(np.array([*dd.values])[:,2])
        cons_df.to_excel(f"../results/{config}_eval_scores.xlsx", index=False)
        
        bleurt = load("bleurt", "BLEURT-20")
        print("Calculating bleurt score...")
        for mname in all_model_names:
            print(mname)
            if f"{mname}_bleurt" in skip_metrics:
                cons_df[f"{mname}_bleurt"] = org_df[f"{mname}_bleurt"]
                continue
            dd = cons_df.apply(lambda x: calc_bleurtscore([x[mname]], [x.ans]), axis=1)
            cons_df[f"{mname}_bleurt"] = list(np.array([*dd.values]))
        bleurt = None
        cons_df.to_excel(f"../results/{config}_eval_scores.xlsx", index=False)

        bertscore = load("bertscore", model_type="microsoft/deberta-xlarge-mnli")
        print("Calculating bertscore...")
        for mname in all_model_names:
            print(mname)
            if f"{mname}_bs_precision" in skip_metrics and f"{mname}_bs_recall" in skip_metrics and f"{mname}_bs_f1" in skip_metrics:
                cons_df[f"{mname}_bs_precision"] = org_df[f"{mname}_bs_precision"]
                cons_df[f"{mname}_bs_recall"] = org_df[f"{mname}_bs_recall"]
                cons_df[f"{mname}_bs_f1"] = org_df[f"{mname}_bs_f1"]
                continue
            cons_df[mname] = cons_df[mname].astype(str)
            dd = cons_df.apply(lambda x: calc_bertscore([x[mname]], [x.ans]), axis=1)
            cons_df[f"{mname}_bs_precision"] = list(np.array([*dd.values])[:,0])
            cons_df[f"{mname}_bs_recall"] = list(np.array([*dd.values])[:,1])
            cons_df[f"{mname}_bs_f1"] = list(np.array([*dd.values])[:,2])
        bertscore = None
        cons_df.to_excel(f"../results/{config}_eval_scores.xlsx", index=False)

        if os.path.isfile(f"./results/{config}_eval_scores.xlsx"):
            org_df = pd.read_excel(f"./results/{config}_eval_scores.xlsx")
            for coli in cons_df.columns:
                if coli in org_df.columns:
                    pass
                else:
                    org_df[coli] = cons_df[coli]
            org_df.to_excel(f"./results/{config}_eval_scores.xlsx", index=False)
            cons_df = org_df.copy()
        else:
            cons_df.to_excel(f"./results/{config}_eval_scores.xlsx", index=False)
        
        print("\n\nDone with evals...")

        for mname in sorted(all_model_names):
            model_metrics = []
            for metric in all_metrics:
                model_metrics.append(round(cons_df[f"{mname}_{metric}"].mean(), 1))
            print(", ".join([mname] + [str(x) for x in model_metrics]))
    return


if __name__ == "__main__":
    run_evals()
## Generate responses for all models for different configurations

## PTABS
python 0A1_llm_generation.py --run_id 3 --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --data_file 'data/scidqa.xlsx' --resp_dir data_resp --config tabs
python 0A1_llm_generation.py --run_id 3 --model_name meta-llama/Meta-Llama-3.1-70B-Instruct --data_file 'data/scidqa.xlsx' --resp_dir data_resp --config tabs

# and so on for other models and configurations ...
# configs are mem, tabs, rag, ft

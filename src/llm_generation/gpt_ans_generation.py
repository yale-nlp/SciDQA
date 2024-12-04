import os
import pandas as pd
import numpy as np
import pickle
import time
import datetime
import json
from openai import AzureOpenAI
from glob import glob
import pathlib 

status_colors = {
    "validating": "\033[1;31;30m",
    "in_progress": "\033[1;31;34m",
    "finalizing": "\033[1;31;32m",
    "completed": "\033[1;31;35m",
}

# Set the Azure API endpoint and key
end_point = ""
client = AzureOpenAI(azure_endpoint=end_point, 
					 api_key="", 
                     api_version="")

def load_metadata(mname="gpt-4o"):
	exp_configs = ['mem', 'tabs', 'rag', "ft"] 
	config_files_list = {conf: [] for conf in exp_configs}
	config_batches_list = {conf: [] for conf in exp_configs}
	submission_responses = {conf: {} for conf in exp_configs}
	batch_status = {}
	
	config_files_list_path = f"./exp_logs/{mname}/config_files_list.pkl"
	config_batches_list_path = f"./exp_logs/{mname}/config_batches_list.pkl"
	submission_responses_path = f"./exp_logs/{mname}/submission_responses.pkl"
	batch_status_path = f"./exp_logs/{mname}/batch_status.pkl"

	if os.path.isfile(config_files_list_path):
		with open(config_files_list_path, "rb") as f:
			config_files_list = pickle.load(f)
	if os.path.isfile(config_batches_list_path):
		with open(config_batches_list_path, "rb") as f:
			config_batches_list = pickle.load(f)
	if os.path.isfile(submission_responses_path):
		with open(submission_responses_path, "rb") as f:
			submission_responses = pickle.load(f)
	if os.path.isfile(batch_status_path):
		with open(batch_status_path, "rb") as f:
			batch_status = pickle.load(f)
	
	for conf in exp_configs:
		if not conf in config_files_list:
			config_files_list[conf] = []
		if not conf in config_batches_list:
			config_batches_list[conf] = []
		if not conf in submission_responses:
			submission_responses[conf] = {}
	return exp_configs, config_files_list, config_batches_list, submission_responses, batch_status


def save_metadata(config_files_list, config_batches_list, submission_responses, batch_status, mname):
	pathlib.Path(f'./exp_logs/{mname}').mkdir(parents=True, exist_ok=True)
	with open(f"./exp_logs/{mname}/config_files_list.pkl", "wb") as f:
		pickle.dump(config_files_list, f)
	with open(f"./exp_logs/{mname}/config_batches_list.pkl", "wb") as f:
		pickle.dump(config_batches_list, f)
	with open(f"./exp_logs/{mname}/submission_responses.pkl", "wb") as f:
		pickle.dump(submission_responses, f)
	with open(f"./exp_logs/{mname}/batch_status.pkl", "wb") as f:
		pickle.dump(batch_status, f)
	return


def update_batch_status(batch_status_dict, mname):
	with open(f"./exp_logs/{mname}/batch_status.pkl", "wb") as f:
		pickle.dump(batch_status_dict, f)
	return


def submit_batch_inputs(mname="gpt-4o"):

	exp_configs, config_files_list, config_batches_list, submission_responses, batch_status = load_metadata(mname)
	
	for config in exp_configs:
		submitted_file_ids = []
		if config == "ft":
			if mname == "gpt-4o":
				submission_files = glob(f"./data/ft/{config}_{mname}_*.jsonl")
			elif mname == "gpt-4o-mini-2":
				submission_files = glob(f"./data/ft/{config}_{mname}_*.jsonl")
		else:
			submission_files = [f"./data/{config}_{mname}.jsonl"]
			
		if len(submission_files) == 0:
			print("Run gpt_util_file_prep.py to prepare the input files.")

		for inp_file in submission_files:
			# Upload a file with a purpose of "batch"
			input_file = client.files.create(file=open(inp_file, "rb"), purpose="batch")
			input_file_id = input_file.id
			print(f"Config: {config} | File id: {input_file_id}")
			submission_responses[config][input_file_id] = input_file.model_dump()
			config_files_list[config].append(input_file_id)
			save_metadata(config_files_list, config_batches_list, submission_responses, batch_status, mname)
			submitted_file_ids.append(input_file_id)
		
		# Wait for file upload to complete. Tracking with status does not seem to work.
		time.sleep(5)
		
		for input_file_id in submitted_file_ids:
			batch_job_metadata = client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"description": f"{config.upper()} generations"})
			batch_id = batch_job_metadata.id
			print(f"Config: {config}; Batch id: {batch_id}")
			config_batches_list[config].append(batch_id)
			submission_responses[config][batch_id] = batch_job_metadata.model_dump()
			batch_status[batch_id] = "waiting"
			save_metadata(config_files_list, config_batches_list, submission_responses, batch_status, mname)

	print("Batch submission complete: ", datetime.datetime.now())
	return

def batch_waiting(mname="gpt-4o"):
	exp_configs, config_files_list, config_batches_list, submission_responses, batch_status = load_metadata(mname)
	while True:
		for config in exp_configs:
			for batch_id in config_batches_list[config]:
				if batch_status[batch_id] == "waiting":
					batch_response = client.batches.retrieve(batch_id)
					status = batch_response.status
					print(status_colors.get(status, ""), end="")
					print(f"Config: {config} | Batch Id: {batch_id} | Status: {status}")
					if status in ("completed", "failed", "canceled"):
						process_batch(config, batch_id, mname)
						batch_status[batch_id] = status
						update_batch_status(batch_status, mname)
		
		if any("waiting" == anybatchstatus for anybatchstatus in batch_status.values()):
			time.sleep(120)
		else:
			print("All batches done processing...")
			break
	return


def process_ft_batch(config, batch_id, mname):
	qa_df = pd.read_excel("../data/scidqa.xlsx")
	batch_response = client.batches.retrieve(batch_id)

	# Save the batch outputs to the file directly first and then to the data_resp repo.
	file_resp = client.files.content(file_id=batch_response.error_file_id)
	with open(os.path.join(f"./exp_logs/{mname}/raw_files/ft", f"{config}_error_file_{batch_id}.txt"), "w") as output_file:
		output_file.write(file_resp.text.strip())

	file_resp = client.files.content(file_id=batch_response.output_file_id)
	raw_responses = file_resp.text.strip().split('\n')
	formatted_json_list = []
	for raw_response in raw_responses:
		try:
			parsed_json = json.loads(raw_response)
			formatted_json_list.append(parsed_json)
		except json.JSONDecodeError as e:
			# Handle the case where a line isn't a valid JSON
			print(f"Error decoding JSON: {e}")
	result_file = os.path.join(f"./exp_logs/{mname}/raw_files/ft", f"{config}_batch_text_results_{batch_id}.json")

	with open(result_file, 'w') as output_file:
		json.dump(formatted_json_list, output_file, indent=5)

	print(f"Config {config} results saved to {result_file}\n")

	with open(result_file, 'r') as file:
		data = json.load(file)

	if os.path.isfile(f'./data_resp/{mname}_{config}.xlsx'):
		partial_ans_df = pd.read_excel(f'./data_resp/{mname}_{config}.xlsx')
		model_ans_list = partial_ans_df[f"{mname}_ans"].tolist()
	else:
		model_ans_list = [""] * qa_df.shape[0]

	# Sort the data by custom_id and add the to QA df to create response df.
	sorted_data = sorted(data, key=lambda x: x["custom_id"])
	for data_instance in sorted_data:
		qid = int(data_instance["custom_id"].split("_")[1].strip())
		try:
			model_ans_list[qid] = data_instance["response"]["body"]["choices"][0]["message"]["content"]
		except:
			print(f"Error in parsing {mname} response for {config} for qid: ", qid)

	with pd.ExcelWriter(f'./data_resp/{mname}_{config}.xlsx') as writer:
		qa_df[f"{mname}_ans"] = model_ans_list
		qa_df.to_excel(writer, sheet_name=mname)
	qa_df.to_pickle(f'./data_resp/{mname}_{config}.pkl')
	
	return


def process_batch(config, batch_id, mname):
	if config == "ft":
		process_ft_batch(config, batch_id, mname)
		return

	qa_df = pd.read_excel("../data/data_v3_jan6.xlsx")
	batch_response = client.batches.retrieve(batch_id)

	# Save the batch outputs to the file directly first and then to the data_resp repo.
	file_resp = client.files.content(file_id=batch_response.error_file_id)
	with open(os.path.join(f"./exp_logs/{mname}/raw_files", f"{config}_error_file.txt"), "w") as output_file:
		output_file.write(file_resp.text.strip())

	file_resp = client.files.content(file_id=batch_response.output_file_id)
	raw_responses = file_resp.text.strip().split('\n')
	formatted_json_list = []
	for raw_response in raw_responses:
		try:
			parsed_json = json.loads(raw_response)
			formatted_json_list.append(parsed_json)
		except json.JSONDecodeError as e:
			# Handle the case where a line isn't a valid JSON
			print(f"Error decoding JSON: {e}")
	result_file = os.path.join(f"./exp_logs/{mname}/raw_files", f"{config}_batch_text_results.json")

	with open(result_file, 'w') as output_file:
		json.dump(formatted_json_list, output_file, indent=5)

	print(f"Config {config} results saved to {result_file}\n")

	with open(result_file, 'r') as file:
		data = json.load(file)

	model_ans_list = [""] * qa_df.shape[0]
	# Sort the data by custom_id and add the to QA df to create response df.
	sorted_data = sorted(data, key=lambda x: x["custom_id"])
	for data_instance in sorted_data:
		qid = int(data_instance["custom_id"].split("_")[1].strip())
		try:
			model_ans_list[qid] = data_instance["response"]["body"]["choices"][0]["message"]["content"]
		except:
			print(f"Error in parsing {mname} response for {config} for qid: ", qid)

	with pd.ExcelWriter(f'./data_resp/{mname}_{config}.xlsx') as writer:
		qa_df[f"{mname}_ans"] = model_ans_list
		qa_df.to_excel(writer, sheet_name=mname)
	qa_df.to_pickle(f'./data_resp/{mname}_{config}.pkl')
	
	return


if __name__ == "__main__":
	for mname in ["gpt-4o", "gpt-4o-mini-2"]:
		submit_batch_inputs(mname)
		batch_waiting(mname)
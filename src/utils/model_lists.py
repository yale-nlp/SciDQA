model2contextlength = {
    "meta-llama/Llama-2-7b-chat-hf": ("llama2_7b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-13b-chat-hf": ("llama2_13b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/blob/main/generation_config.json"), 
    "meta-llama/Llama-2-70b-chat-hf": ("llama2_70b_chat", 4096, "https://huggingface.co/meta-llama/Llama-2-70b-chat-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-7b-hf": ("llama2_7b", 4096, "https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-13b-hf": ("llama2_13b", 4096, "https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/generation_config.json"),
    "meta-llama/Llama-2-70b-hf": ("llama2_70b", 4096, "https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/generation_config.json"),
    "mistralai/Mistral-7B-v0.1": ("mistral_7b", 8000, "https://aws.amazon.com/blogs/machine-learning/mistral-7b-foundation-models-from-mistral-ai-are-now-available-in-amazon-sagemaker-jumpstart/#:~:text=Mistral%207B%20has%20an%208%2C000,at%20a%207B%20model%20size."),
    "mistralai/Mistral-7B-Instruct-v0.1": ("mistral_7b_instruct", 8000, "https://www.secondstate.io/articles/mistral-7b-instruct-v0.1/"),
    "mistralai/Mistral-7B-Instruct-v0.2": ("mistral_7b_instruct_v2", 8000, "https://www.secondstate.io/articles/mistral-7b-instruct-v0.1/"), 
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
    "meta-llama/Meta-Llama-3-8B-Instruct": ("llama3_8b", 8192, ""),
    # We can only fit 4096 tokens
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ("llama31_8b", 4096, ""),
    "meta-llama/Meta-Llama-3.1-70B-Instruct": ("llama31_70b", 4096, ""),
    "Qwen/Qwen2.5-1.5B-Instruct": ("qwen_1pt5B_IT", 4096, ""),
    "Qwen/Qwen2.5-7B-Instruct": ("qwen_7B_IT", 4096, ""),
    "Gemini": ("gemini", 128000, ""),
    "GPT4": ("gpt4", 128000, ""),
    "GPT-4o": ("gpt4o", 128000, ""),
    "GPT-4o-mini-2": ("gpt4omini", 128000, "")
}

model_name_map = {"meta-llama/Llama-2-7b-chat-hf": "llama2_7b_chat", "meta-llama/Llama-2-13b-chat-hf": "llama2_13b_chat", "meta-llama/Llama-2-70b-chat-hf": "llama2_70b_chat", "meta-llama/Llama-2-7b-hf": "llama2_7b", "meta-llama/Llama-2-13b-hf": "llama2_13b", "meta-llama/Llama-2-70b-hf": "llama2_70b", "mistralai/Mistral-7B-v0.1": "mistral_7b", "mistralai/Mistral-7B-Instruct-v0.1": "mistral_7b_instruct", "mistralai/Mistral-7B-Instruct-v0.2": "mistral_7b_instruct_v2", "mistralai/Mixtral-8x7B-v0.1": "mistral_8_7b", "mistralai/Mixtral-8x7B-Instruct-v0.1": "mistral_8_7b_instruct", "lmsys/vicuna-13b-v1.5": "vicuna_13b", "lmsys/vicuna-13b-v1.5-16k": "vicuna_13b_16k", "lmsys/longchat-7b-v1.5-32k": "longchat_7b_32k", "lmsys/vicuna-7b-v1.5": "vicuna_7b", "lmsys/vicuna-7b-v1.5-16k": "vicuna_7b_16k", "HuggingFaceH4/zephyr-7b-beta": "zephyr_7b_beta", "tiiuae/falcon-7b": "falcon_7b", "tiiuae/falcon-7b-instruct": "falcon_7b_instruct", "tiiuae/falcon-40b": "falcon_40b", "facebook/galactica-6.7b": "galactica_7b", "facebook/galactica-30b": "galactica_30b", "microsoft/phi-2": "ms_phi2_3b", "google/gemma-2b-it": "gemma_2b_it", "google/gemma-2b": "gemma_2b", "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b", "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama31_8b",  "meta-llama/Meta-Llama-3.1-70B-Instruct": "llama31_70b"}
from PROMPTS import PROMPTS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import os
import sys
import json
import argparse
from transformers import BitsAndBytesConfig
from copy import deepcopy
from string import Template

def main():
    parser = argparse.ArgumentParser()
    
    # Argument
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument("--reasoning", action="store_true", default=False)
    parser.add_argument("--dataset-path", default="./data/RegexPSPACE.jsonl", type=str)
    parser.add_argument("--fewshot-path", default="./data/RegexPSPACE_fewshot.jsonl", type=str)
    parser.add_argument("--task", choices=["minimization", "equivalence"], type=str)
    parser.add_argument("--shot", choices=["five", "zero"], default="zero", type=str)
    parser.add_argument("--max-think-tokens", default=4096, type=int)
    parser.add_argument("--max-answer-tokens", default=1024, type=int)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--top-p", default=None, type=float)
    parser.add_argument("--gpt-oss-mode", choices=["high", "medium", "low"], default="medium")
    parser.add_argument("--start-idx", default=0, type=int)
    parser.add_argument("--end-idx", default=2147483647, type=int)
    
    args = parser.parse_args()

    model_name = args.model_name
    reasoning = args.reasoning
    dataset_path = args.dataset_path
    fewshot_path = args.fewshot_path
    task = args.task
    shot = args.shot
    max_think_tokens = args.max_think_tokens
    max_answer_tokens = args.max_answer_tokens
    do_sample = args.do_sample
    temperature = args.temperature
    top_p = args.top_p
    gpt_oss_mode = args.gpt_oss_mode
    start_idx = args.start_idx
    end_idx = args.end_idx
    
    if start_idx >= end_idx:
        print("Start index is larger or equal to End index!")
        exit(1)
    
    # Loading Dataset
    with open(dataset_path, 'r') as f:
        data = [json.loads(i) for i in f.readlines()]
        
    # Setting Output Path
    path = f"./result/{task}/{shot}/"
    os.makedirs(path, exist_ok=True)
    
    # Loading Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )
    if "gpt-oss" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            quantization_config=quantization_config,
            device_map="auto"
        )
    model.eval()

    # Setting Config and Variables for Reasoning
    if reasoning:
        thinking_generation_config = deepcopy(model.generation_config)
        thinking_generation_config.max_new_tokens = max_think_tokens
        if "gpt-oss" in model_name:
            think_end_token = "<|end|>"
        else:
            think_end_token = "</think>"
        if len(tokenizer.encode(think_end_token, add_special_tokens=False))==1:
            thinking_generation_config.eos_token_id = tokenizer.encode(think_end_token, add_special_tokens=False)[0]
        else:
            thinking_generation_config.stop_strings = think_end_token

    # Setting Config for Sampling
    model.generation_config.do_sample = do_sample
    if do_sample:
        if temperature != None:
            model.generation_config.temperature = temperature
        if top_p != None:
            model.generation_config.top_p = top_p

    # Output Path
    result_path = path+model_name.split("/")[1]
    if "gpt-oss" in model_name.lower():
        result_path += f"_{gpt_oss_mode}"
    result_path += f"_{reasoning}"
    if do_sample:
        result_path += f"_{model.generation_config.temperature}_{model.generation_config.top_p}"
    if start_idx > 0:
        result_path += f"_{start_idx}"
    result_path += ".jsonl"
    
    # Setting Maximum Answer Tokens
    answer_generation_config = deepcopy(model.generation_config)
    answer_generation_config.max_new_tokens = max_answer_tokens

    # Loading Prompts
    prompt = Template(PROMPTS["_".join([task, shot, "prompt"])])
    if shot=="five":
        with open(fewshot_path, 'r') as f:
            fewshot_examples = [json.loads(i) for i in f.readlines()][:5]
        if task == "minimization":
            prompt = Template(
                prompt.safe_substitute(
                    example1_input_regex = fewshot_examples[0]["query"],
                    example1_output = fewshot_examples[0]["minimized_regex"],
                    example2_input_regex = fewshot_examples[1]["query"],
                    example2_output = fewshot_examples[1]["minimized_regex"],
                    example3_input_regex = fewshot_examples[2]["query"],
                    example3_output = fewshot_examples[2]["minimized_regex"],
                    example4_input_regex = fewshot_examples[3]["query"],
                    example4_output = fewshot_examples[3]["minimized_regex"],
                    example5_input_regex = fewshot_examples[4]["query"],
                    example5_output = fewshot_examples[4]["minimized_regex"]
                )
            )
        elif task == "equivalence":
            prompt = Template(
                prompt.safe_substitute(
                    example1_input_regex1 = fewshot_examples[0]["query"],
                    example1_input_regex2 = fewshot_examples[0]["equivalent_regex"],
                    example1_output = True,
                    example2_input_regex1 = fewshot_examples[1]["query"],
                    example2_input_regex2 = fewshot_examples[1]["not_equivalent_regex"],
                    example2_output = False,
                    example3_input_regex1 = fewshot_examples[2]["query"],
                    example3_input_regex2 = fewshot_examples[2]["equivalent_regex"],
                    example3_output = True,
                    example4_input_regex1 = fewshot_examples[3]["query"],
                    example4_input_regex2 = fewshot_examples[3]["not_equivalent_regex"],
                    example4_output = False,
                    example5_input_regex1 = fewshot_examples[4]["query"],
                    example5_input_regex2 = fewshot_examples[4]["equivalent_regex"],
                    example5_output = True,
                )
            )
        else:
            print(f"The task \"{task}\" is not defined.")
            exit(1)

    # Trimming Already Existing Files
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            lines = f.readlines()
        temp = [json.loads(i) for i in lines[:-1]]
        try:
            last_line = json.loads(lines[-1])
            temp.append(last_line)
        except:
            pass
        if task == "equivalence" and len(temp)<2:
            os.system(f"rm {result_path}")
        else:
            if task == "equivalence" and temp[-1]["idx"] != temp[-2]["idx"]:
                temp.pop()
            with open(result_path, "w") as f:
                for i in temp:
                    json.dump(i, f)
                    f.write("\n")
            start_idx = max(start_idx, temp[-1]["idx"]+1)
    
    # Start Inferencing
    with open(result_path, 'a') as f:
        for i, x in enumerate(tqdm(data[start_idx:end_idx])):
            # Creating Queries
            if task == "minimization":
                queries = [(prompt.substitute(
                    regex = x["query"]
                ), x["minimized_regex"], x["query"])]
            elif task == "equivalence":
                queries = [
                    (prompt.substitute(
                        regex1 = x["query"],
                        regex2 = x["equivalent_regex"],
                    ), True, x["query"]),
                    (prompt.substitute(
                        regex1 = x["query"],
                        regex2 = x["not_equivalent_regex"],
                    ), False, x["query"])
                ]
            else:
                print(f"The task \"{task}\" is not defined.")
                exit(1)
            
            for query, answer, query_regex in queries:
                # Applying Chat Template
                if "exaone" in model_name.lower():
                    if reasoning:
                        input_template = tokenizer.apply_chat_template([{"role":"user","content":query}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    else:
                        input_template = tokenizer.apply_chat_template([{"role":"user","content":query}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                elif "gpt-oss" in model_name.lower():
                    input_template = tokenizer.apply_chat_template([{"role":"user", "content":query}], reasoning_effort=gpt_oss_mode, tokenize=False, add_generation_prompt=True)
                else:
                    input_template = tokenizer.apply_chat_template([{"role":"user","content":query}], tokenize=False, add_generation_prompt=True)
                
                # Tokenizing
                inputs = tokenizer(input_template, add_special_tokens=False, return_tensors="pt").to('cuda')
                query_length = len(inputs["input_ids"][0])
                
                # Reasoning Inference
                if reasoning:
                    if len(tokenizer.encode(think_end_token, add_special_tokens=False))==1:
                        output_ids = model.generate(**inputs, generation_config=thinking_generation_config)[0]
                        output_string = tokenizer.decode(output_ids[query_length:])
                    else:
                        output_ids = model.generate(**inputs, generation_config=thinking_generation_config, tokenizer=tokenizer)[0]
                        output_string = tokenizer.decode(output_ids[query_length:])
                    
                    inputs = tokenizer(input_template+output_string.split(think_end_token)[0]+think_end_token, add_special_tokens=False, return_tensors="pt").to('cuda')
                    thinking_length = len(inputs["input_ids"][0]) - query_length

                # Answer Inference
                output_ids = model.generate(**inputs, generation_config=answer_generation_config, tokenizer=tokenizer)[0]
                if reasoning:
                    answer_length = len(output_ids) - query_length - thinking_length
                else:
                    answer_length = len(output_ids) - query_length
                
                # Resulting Output
                result = {
                    "idx":x["idx"],
                    "query_regex":query_regex,
                    "answer":answer,
                    "prompt":tokenizer.decode(output_ids[:query_length]),
                }
                if reasoning:
                    result["generated_thinking"] = tokenizer.decode(output_ids[query_length:query_length+thinking_length])
                result["generated_answer"] = tokenizer.decode(output_ids[-answer_length:])

                # Saving Output
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
            
if __name__ == "__main__":
    main()
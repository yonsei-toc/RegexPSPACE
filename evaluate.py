from FAdo.reex import *
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
import os
import time
import json
import numpy as np
import argparse


def build_reversed_KMP_table(input_string: str):
    n = len(input_string)
    table = [0] * n
    for i in range(1, n):
        j = table[i - 1]
        while j > 0 and input_string[i] != input_string[j]:
            j = table[j - 1]
        if input_string[i] == input_string[j]:
            j += 1
        table[i] = j
    return table

def find_repeating_suffix(input_string: str):
    n = len(input_string)
    table = build_reversed_KMP_table(input_string[::-1])
    results = []

    # Check whether a suffix is repeated
    for idx in range(n//2):
        suffix_length = idx + 1
        min_phrase_length = suffix_length - table[idx]
        if suffix_length % min_phrase_length == 0:
            count = suffix_length // min_phrase_length
            phrase = input_string[n-suffix_length:n-suffix_length+min_phrase_length]
            results.append((phrase, count, suffix_length))

    # Return the suffix with maximum repetition
    if results:
        phrase, count, length = max(results, key=lambda x: (x[1], x[2]))
        return phrase, count
    else:
        return input_string, 1

def geometric_mean(x):
    a = np.log(x)
    return np.exp(a.mean())

def findbox(string):
    # Parsing the LLMs' responses with heuristics
    if not "boxed" in string:
        for keyword in ["answer:", "answer**"]:
            for line in string.lower().split("\n")[::-1]:
                if keyword in line:
                    words = line.split()
                    for j in range(len(words)):
                        if keyword in words[j] and len(words) > j+1:
                            return " ".join(words[(j+1):]).split("<｜end▁of▁sentence｜>")[0].split("<|end_of_sentence|>")[0].split("<eos>")[0].replace("\\", "").replace(".", "").replace(",", "")
        for phrase in ["minimized regex", "final simplifed regex", "final regex" "minimal regex", "minimal form", "answer"]:
            for line in string.lower().split("\n")[::-1]:
                if phrase in line:
                    words = line.split()
                    for j in range(len(words)):
                        if "is" in words[j]:
                            return " ".join(words[(j+1):]).split("<｜end▁of▁sentence｜>")[0].split("<|end_of_sentence|>")[0].split("<eos>")[0].replace("\\", "").replace(".", "").replace(",", "")
                        if ":" in words[j]:
                            return " ".join(words[(j+1):]).split("<｜end▁of▁sentence｜>")[0].split("<|end_of_sentence|>")[0].split("<eos>")[0].replace("\\", "").replace(".", "").replace(",", "")
        return string.split("\n")[-1]
    
    # Parsing the LLMs' responses with boxed tags
    last_box = string.split("boxed{")[-1]
    stack = ['{']
    answer=""
    for i in last_box:
        if i in "{([<":
            stack.append(i)
        elif i=="}":
            if stack[-1]=="{":
                stack.pop()
                if len(stack) == 0:
                    break
            else:
                return last_box
        elif i==")":
            if stack[-1]=="(":
                stack.pop()
            else:
                return last_box
        elif i=="]":
            if stack[-1]=="[":
                stack.pop()
            else:
                return last_box
        elif i==">":
            if stack[-1]=="<":        
                stack.pop()
            else:
                return last_box
        answer += i
    if "text" in answer:
        last_box = string.split("text{")[-1]
        stack = ['{']
        answer=""
        for i in last_box:
            if i in "{([<":
                stack.append(i)
            elif i=="}":
                if stack[-1]=="{":
                    stack.pop()
                    if len(stack) == 0:
                        break
                else:
                    return last_box
            elif i==")":
                if stack[-1]=="(":
                    stack.pop()
                else:
                    return last_box
            elif i=="]":
                if stack[-1]=="[":
                    stack.pop()
                else:
                    return last_box
            elif i==">":
                if stack[-1]=="<":        
                    stack.pop()
                else:
                    return last_box
            answer += i
    return answer

def main():
    parser = argparse.ArgumentParser()
    
    # Argument
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type=str)
    parser.add_argument("--dataset-path", default="./data/RegexPSPACE.jsonl", type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--task", choices=["minimization", "equivalence"], type=str)
    parser.add_argument("--max-answer-tokens", default=1024, type=int)
    
    args = parser.parse_args()
    
    # Loading dataset
    with open(args.dataset_path, 'r') as f:
        dataset = [json.loads(i) for i in f.readlines()]

    # Building regexes
    dataset_regexes = [str2regexp(i["query"], sigma=set(list("abcd")), strict=True) for i in dataset]

    # Loading LLM output
    with open(args.output_path, 'r') as f:
        outputs = [json.loads(i) for i in f.readlines()]
    
    if args.task=="minimization":
        assert(len(dataset) == len(outputs))
    elif args.task=="equivalence":
        assert(2*len(dataset) == len(outputs))
    
    print("="*80)
    print(f"Evaluating {args.output_path}")
    print("="*80)
    evaluate(outputs, dataset, dataset_regexes, args.task)
    print("="*80)
    analyze_failure(outputs, dataset, dataset_regexes, args.task, args.model_name, args.max_answer_tokens)
    print("="*80)

def evaluate(outputs, dataset, dataset_regexes, task):
    if task == "minimization":
        # Evaluating minimization
        minimality = 0
        equivalence = 0
        tree_length_ratio = []
        for idx, output in enumerate(outputs):
            # Postprocessing outputs
            parsed_answer = findbox(output["generated_answer"])
            temp = parsed_answer.replace(" ", "").replace("^", "").replace("|", "+").replace("{", "").replace("}", "")
            # Checking validity
            try:
                if len(temp) > 2*len(output["query_regex"]):
                    raise Exception('Length is increased!')
                regex = str2regexp(temp, sigma=set(list("abcd")), strict=True)
                valid = True
            except:
                valid = False

            if valid:
                # Checking equivalence
                if regex.equivalentP(dataset_regexes[idx]):
                    equivalence+=1
                    # Checking minimality
                    if regex.treeLength() == dataset[idx]["minimized_tree_length"]:
                        minimality+=1
                    ratio = regex.treeLength()/dataset_regexes[idx].treeLength()
                    if ratio > 1:
                        tree_length_ratio.append(1.0)
                    else:
                        tree_length_ratio.append(ratio)
                    continue
            tree_length_ratio.append(1.0)

        print(f"Minimality: {minimality/len(dataset)*100:.4f}")
        print(f"Equivalence: {equivalence/len(dataset)*100:.4f}")
        print(f"Length Ratio: {geometric_mean(tree_length_ratio)*100:.4f}")

    elif task=="equivalence":
        # Evaluating equivalence
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        Fail = 0
        for idx, output in enumerate(outputs):
            # Postprocessing outputs
            parsed_answer = findbox(output["generated_answer"]).lower()
            # Checking the answer
            if "true" in parsed_answer:
                if output["answer"]:
                    TP += 1
                else:
                    FP += 1
            elif "false" in parsed_answer:
                if output["answer"]:
                    FN += 1
                else:
                    TN += 1
            else:
                Fail += 1
        accuracy = (TP+TN) / len(outputs)
        f1_score = 2*TP / (2*TP+FP+FN)
        failure_rate = Fail / len(outputs)

        print(f"Accuracy: {accuracy*100:.4f}")
        print(f"F1-score: {f1_score*100:.4f}")
        print(f"Failure Rate: {failure_rate*100:.4f}")

def analyze_failure(outputs, dataset, dataset_regexes, task, model_name, max_answer_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if task == "minimization":
        # Evaluating minimization
        minimal = 0
        not_minimal_but_equivalent = 0
        not_equivalent_but_valid = 0
        stopped_but_invalid = 0
        fail_with_repetition = 0
        max_token_limit = 0
        for idx, output in enumerate(outputs):
            # Postprocessing outputs
            parsed_answer = findbox(output["generated_answer"])
            temp = parsed_answer.replace(" ", "").replace("^", "").replace("|", "+").replace("{", "").replace("}", "")
            # Checking validity
            try:
                if len(temp) > 2*len(output["query_regex"]):
                    raise Exception('Length is increased!')
                regex = str2regexp(temp, sigma=set(list("abcd")), strict=True)
                valid = True
            except:
                valid = False

            if valid:
                # Checking equivalence
                if regex.equivalentP(dataset_regexes[idx]):
                    # Checking minimality
                    if regex.treeLength() == dataset[idx]["minimized_tree_length"]:
                        minimal += 1
                    else:
                        not_minimal_but_equivalent += 1
                else:
                    not_equivalent_but_valid += 1
            else:
                # Calculating repetition
                phrase, count = find_repeating_suffix(output["generated_answer"])
                if count > 2 and len(phrase)>10:
                    fail_with_repetition += 1
                else:
                    # Checking if the maximum number of tokens are generated
                    if len(tokenizer(output["generated_answer"])["input_ids"]) == max_answer_tokens:
                        max_token_limit += 1
                    else:
                        stopped_but_invalid += 1

        print(f"Minimal: {minimal/len(outputs)*100:.4f}")
        print(f"Equivalent: {not_minimal_but_equivalent/len(outputs)*100:.4f}")
        print(f"Valid: {not_equivalent_but_valid/len(outputs)*100:.4f}")
        print(f"Invalid: {stopped_but_invalid/len(outputs)*100:.4f}")
        print(f"Repeated: {fail_with_repetition/len(outputs)*100:.4f}")
        print(f"Stopped: {max_token_limit/len(outputs)*100:.4f}")

    elif task =="equivalence":
        # Evaluating equivalence
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        stopped_but_invalid = 0
        fail_with_repetition = 0
        max_token_limit = 0
        for idx, output in enumerate(outputs):
            # Postprocessing outputs
            parsed_answer = findbox(output["generated_answer"]).lower()
            # Checking the answer
            if "true" in parsed_answer:
                if output["answer"]:
                    TP+=1
                else:
                    FP+=1
            elif "false" in parsed_answer:
                if output["answer"]:
                    FN+=1
                else:
                    TN+=1
            else:
                # Calculating repetition
                phrase, count = find_repeating_suffix(output["generated_answer"])
                if count > 2 and len(phrase)>10:
                    fail_with_repetition+=1
                else:
                    # Checking if the maximum number of tokens are generated
                    if len(tokenizer(output["generated_answer"])["input_ids"]) == max_answer_tokens:
                        max_token_limit+=1
                    else:
                        stopped_but_invalid+=1

        print(f"True Positive: {TP/len(outputs)*100:.4f}")
        print(f"True Negative: {TN/len(outputs)*100:.4f}")
        print(f"False Positive: {FP/len(outputs)*100:.4f}")
        print(f"False Negative: {FN/len(outputs)*100:.4f}")
        print(f"Invalid: {stopped_but_invalid/len(outputs)*100:.4f}")
        print(f"Repeated: {fail_with_repetition/len(outputs)*100:.4f}")
        print(f"Stopped: {max_token_limit/len(outputs)*100:.4f}")

if __name__ == "__main__":
    main()
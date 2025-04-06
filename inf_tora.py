import random
import os
import argparse
import time
import torch
from datetime import datetime
from tqdm import tqdm
import sys, os
#sys.path.append(os.path.abspath('..'))
from evaluate_tora import evaluate
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import parse_question, extract_program_output, parse_ground_truth
from utils.data_loader import load_data
from python_executor import PythonExecutor
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data/tora/examples.jsonl", type=str)
    parser.add_argument("--model_name", default="tora-code-1.3b", type=str)
    parser.add_argument("--model_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tora", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_train_prompt_format", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_dir, data_name, split=None):
    examples = load_data(data_file_path=data_dir, data_name=data_name, split=split)
    return examples 


def encode_with_messages_format(example):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    return example_text


def extract_code_block(text, start_marker='```python', end_marker='```'):
    """
    从文本中提取位于特定标记之间的代码块。
    
    参数:
        text (str): 包含代码块的原始文本
        start_marker (str): 代码块开始的标记，默认为'```python'
        end_marker (str): 代码块结束的标记，默认为'```'
        
    返回:
        str: 提取出的代码块内容，不包含开始和结束标记
    """
    start_index = text.find(start_marker)
    if start_index == -1:
        return ""  # 没找到开始标记
        
    # 跳过开始标记
    start_index += len(start_marker)
    
    end_index = text.find(end_marker, start_index)
    if end_index == -1:
        return ""  # 没找到结束标记
        
    return text[start_index:end_index].strip()
    


def main(args):
    tora_examples = prepare_data('./data/tora/examples.jsonl', 'tora', None)
    #print(tora_examples[0])
    math_examples = prepare_data('./data/math/train.jsonl', 'gsm8k', None)
    #print(math_examples[0])

    
    for test_instance in tora_examples:
        #print(test_instance['id'])
        ti_str = test_instance['id']
        ti_cot = test_instance['gt_cot']
        ti_str_train_id = int(ti_str.split('-')[2])
        #print('===')
        #print(math_examples[ti_str_train_id]['problem'])
        #print('----')
        #print(ti_cot)
        # append
        test_instance['problem'] = math_examples[ti_str_train_id]['problem']
        test_instance['solution'] = math_examples[ti_str_train_id]['solution']

    # init python executor
    executor = PythonExecutor(get_answer_from_stdout=True)

    examples = tora_examples[:5]
    # print(examples)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']
        message_str = encode_with_messages_format(example)

        # extract code from jsonl
        code_str = extract_code_block(message_str)
        #code_str = extract_program(message_str)
        code_result = executor.apply(code_str)
        # parse question and answer
        example['question'] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        print('=== Question ID ===', idx)
        print('\t'+example['question'])
        # add CoT prompts to each question
        full_prompt = construct_prompt(args, example)
        #print('\t----gt+cot')
        #print('\t',gt_cot)
        print('\t---code',code_str)
        print('\t--gt ans ',gt_ans)
        print('\t--code ans', code_result)
        print()
        
        print(example.keys())
        sample = {'idx': idx, 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt, 'code': code_str}
        sample['solution'] = example['solution']
        sample['question'] = example['question'] 
        sample['level'] = example['level'] 

        # add remain fields
        # for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
        #     'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
        #     if key in example:
        #         sample[key] = example[key]
        # print(sample.keys())
        samples.append(sample)
        print()  
        
    print("dataset:", args.data_name, "samples:", len(samples))
    #print(samples)

    # max_func_call = 1 #if args.prompt_type in ['cot', 'pal'] else 4
    # stop_tokens = ["</s>", "```output"]
    # print(end_prompts)
    # print('-------  xxxxxxxx')
    # # sort by idx
    # end_prompts = sorted(end_prompts, key=lambda x: x[0])
    # ans_split = "<|assistant|>" if args.use_train_prompt_format else "Question:"
    # codes = [prompt.split(ans_split)[-1].strip() for _, prompt in end_prompts]

    # # extract preds
    # results = [run_execute(executor, code, args.prompt_type) for code in codes]
    # time_use = time.time() - start_time

    # # put results back to examples
    # all_samples = []
    # for i, sample in enumerate(samples):
    #     code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
    #     result = results[i*args.n_sampling: (i+1)*args.n_sampling]
    #     preds = [item[0] for item in result]
    #     reports = [item[1] for item in result]

    #     sample.pop('prompt')
    #     sample.update({'code': code, 'pred': preds, 'report': reports})
    #     all_samples.append(sample)

    # # add processed samples
    # all_samples.extend(processed_samples)
    # save_jsonl(all_samples, out_file)

    # result_str = evaluate(samples=all_samples, data_name=args.data_name, prompt_type=args.prompt_type, execute=True)
    # result_str += f"\nTime use: {time_use:.2f}s"
    # time_str = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    # result_str += f"\nTime use: {time_str}"

    # with open(out_file.replace(".jsonl", f"_{args.prompt_type}.metrics"), "w") as f:
    #     f.write(result_str)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)

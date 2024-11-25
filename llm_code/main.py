import ipdb.stdout
import ollama
import json
import ipdb
from prompt import *
import random
from string import Template
import argparse
import random
import numpy as np
from alive_progress import alive_bar

def run_ollama(prompt,  engine="qwen2"):
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)

    response = ollama.chat(model=engine, 
                            messages=messages, 
                            options={
                            "temperature":0.0, # default 0.8 
                            },
                            keep_alive = '1m', #内存中存在一分钟
                            )
    print(response['message']['content'])
    result = response['message']['content']

    return result


def tokens_to_string(tokens):
    # 定义不需要前置空格的标点符号
    no_space_before = {',', '.', ')', '/'}
    # 定义不需要后置空格的标点符号
    no_space_after = {'(', '/'}
    
    result = tokens[0]  # 添加第一个token
    
    for i in range(1, len(tokens)):
        current_token = tokens[i]
        previous_token = tokens[i-1]
        
        # 判断是否需要添加空格
        if (current_token not in no_space_before) and (previous_token not in no_space_after):
            result += ' '
        
        result += current_token
    
    return result


def read_json_by_line(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line.strip())
                
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in line: {line}")
                print(f"Error message: {e}")
    return data

def read_dict_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        original_dict = json.load(f)
        # 创建反转的字典
        inverted_dict = {v: k for k, v in original_dict.items()}
        return original_dict, inverted_dict
    

def list_to_numbered_string(lst):
    result = []
    for i, item in enumerate(lst, 1):  # enumerate从1开始计数
        result.append(f"{i}.{item}")
    return ' '.join(result)


def random_read_lines1(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 随机打乱行的顺序
        random.shuffle(lines)
        return lines


def write_list_to_json1(lst, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(lst, f)

if __name__ == '__main__':
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLM_type", "-lt", type=str,
                        default="qwen2", help="base LLM model.", choices=['qwen2', 
                                                                          'llama3.1:8b',
                                                                          'qwen:72b',
                                                                          'llama2:13b',
                                                                          'mistral',
                                                                          'glm4',
                                                                          'qwen2.5:32b',
                                                                          'gpt-3.5-turbo',
                                                                          'gpt-4o-mini',
                                                                          'phi3:14b',
                                                                          'llama2',
                                                                          'qwen2:72b'])
    args = parser.parse_args()


    lines = random_read_lines1('./Wiki80/test.txt')
    # 获取所有的关系
    relation_dict, relation_dict_inverted = read_dict_from_json('./Wiki80/rel2wiki.json')
    relation_list = list(relation_dict.keys())
    relation_str = list_to_numbered_string(relation_list)
    result_list = []
    with alive_bar(len(lines),title='wiki80',bar='classic') as bar:
        for idx, line in enumerate(lines):
            try:
                json_obj = json.loads(line.strip())
                token_string = tokens_to_string(json_obj['token'])
                original_token = json_obj['token']
                head_entity_index = json_obj['h']['pos']
                tail_entity_index = json_obj['t']['pos']
                h_en = tokens_to_string(original_token[head_entity_index[0]:head_entity_index[1]])
                t_en = tokens_to_string(original_token[tail_entity_index[0]:tail_entity_index[1]])
                true_relation = relation_dict_inverted[json_obj['relation']]
                # 提取出关系
                relation = json_obj['relation']

                # print(token_string)
                # print(h_en)
                # print(t_en)
                print(f"true:{true_relation}")

                prompt= Template(prompt_keff).substitute(question=token_string, entity1=h_en, entity2=t_en)
                response = run_ollama(prompt, args.LLM_type)
                
                # 然后统计结果
                result_dict = {
                    "true":true_relation,
                    "llm_response" : response
                }
                result_list.append(result_dict)
                # break
                bar()
            except :
                continue
        write_list_to_json1(result_list, f'{args.LLM_type}_result.json')
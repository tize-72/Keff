from MapTuning import train
from formatter.basic_formatter import bert_formatter
import json
import torch
import os
from tqdm import tqdm
from tools.seed import seed_everything
from tools.gpu_tool import set_gpu
from mapper import AffineMapper_KEFF
# import ipdb ;ipdb.set_trace()
from all_embedding import KE_Embedding, get_PLMembeddings
# import ipdb ;ipdb.set_trace()
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer   
from MapTuning_add_test import train_new
import argparse
from config_parser import create_config
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
import pynvml


def get_most_free_memory_gpu():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    max_free_memory = 0
    best_gpu = 0
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        free_memory = memory_info.free
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_gpu = i
    
    pynvml.nvmlShutdown()
    return best_gpu





class MapTuing_Wikipedia_formatter(bert_formatter) :
    def __init__(self, config) :
        super().__init__(config)

    def get_format(self, mention, Qid) :
        original_mention = mention
        Qword = None
        if Qid not in self.keembedding.ent_map :
            Qid = None
        else :
            Qword = "hashdonothackme" + Qid.lower() + "hashdonothackme"

        MASKS_original = ["[MASK]"] * len(self.tokenizer.tokenize(" ".join(mention)))
        MASKS = ["[MASK]"] * len(self.tokenizer.tokenize(" ".join(mention)))
        if Qid is None :
            pass
        else :
            mention = [Qword] + ["/"] + mention
            MASKS = [Qword] + ["/"] + MASKS_original
        return [" "] + mention + [" "], [" "] + MASKS + [" "], \
            (Qid is not None), MASKS_original, [" "]+[Qword]+[" "], [" "]+original_mention+[" "]


    def template2sentence(self, text, h_word, t_word, pos1, pos2) :
        if pos1[0] < pos2[0] :
            text = text[: pos1[0]] + h_word + text[pos1[1] : pos2[0]] + t_word + text[pos2[1] :]
        else :
            text = text[: pos2[0]] + t_word + text[pos2[1] : pos1[0]] + h_word + text[pos1[1] :]
        return " ".join(text)

    def template2sentence_change_location(self, text, h_word, t_word, pos1, pos2) :
        if pos1[0] < pos2[0] :
            text = text[: pos1[0]] + h_word + text[pos1[1] : pos2[0]] + t_word + text[pos2[1] :]
        else :
            text = text[: pos2[0]] + t_word + text[pos2[1] : pos1[0]] + h_word + text[pos1[1] :]
        return " ".join(text)
    
    def Process(self, mode) :
        data_path = self.config.get("data", "data_path")
        with open(os.path.join(data_path, "wiki20m_{}.txt".format(mode)), "r", encoding = "utf-8") as fin :
            data = [json.loads(line) for line in fin]
        
        inputs_all = []
        inputs = []
        tokens_text_list = []
        for instance in tqdm(data, ncols=120, colour='green') :
            h_list, t_list = instance['h']['name'].split(' '), instance['t']['name'].split(' ')
            text, pos1, pos2 = instance["token"], instance["h"]["pos"], instance["t"]["pos"]
            text = [x.lower() for x in text]
            h_labels, h_MASKS, h_have, h_mask, h_qword, h_original_mention = self.get_format(text[pos1[0] : pos1[1]], instance["h"]["id"])
            h_token_id_list = [-100]*64
            h_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" ".join(text[pos1[0] : pos1[1]])))
            for i in range(len(h_token_id)):h_token_id_list[i] = h_token_id[i]
            t_labels, t_MASKS, t_have, t_mask, t_qword, t_original_mention = self.get_format(text[pos2[0] : pos2[1]], instance["t"]["id"])
            t_token_id_list = [-100]*64
            t_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" ".join(text[pos2[0] : pos2[1]])))
            for i in range(len(t_token_id)):t_token_id_list[i] = t_token_id[i]
            if (not h_have) and (not t_have) :
                continue
    

            Labels = self.tokenizer.tokenize(self.template2sentence(text, h_labels, t_labels, pos1, pos2))
            # import ipdb ;ipdb.set_trace()
            if len(Labels) > self.max_length - 2 :
                continue
            for a in (False, True) :
                for b in (False, True) :
                    if a and b :
                        continue
                    if not a and not b :
                        continue
                    tokens = self.tokenizer.tokenize(self.template2sentence(text, 
                                                                                h_original_mention if a else h_mask, 
                                                                                t_original_mention if b else t_mask,
                                                                                pos1, pos2))

                    labels = self.tokenizer.tokenize(self.template2sentence(text, 
                                                                                h_original_mention, 
                                                                                t_original_mention,
                                                                                pos1, pos2))
                    # import ipdb ;ipdb.set_trace()
                    for (i, _token) in enumerate(tokens) :
                        if _token != "[MASK]" :
                            assert(_token == labels[i])
                            labels[i] = -100
                        else :
                            labels[i] = self.tokenizer.vocab[labels[i]]
                    
                    # 这里补全一个，使得整体凑为65
                    tokens_text = ["[CLS]"] + tokens + ["[SEP]"]
                    tokens_text_list.append(tokens_text)
                    labels = [-100] + labels + [-100]
                    assert(len(tokens_text) == len(labels))
                    tokens = self.tokenizer.convert_tokens_to_ids(tokens_text)
                    mask = [1] * len(tokens)
                    padding = [0] * (self.max_length - len(tokens))
                    labels += [-100] * (self.max_length - len(tokens))
                    tokens += padding
                    mask += padding
                    assert(len(tokens) == len(labels))
                    if a:
                        inputs.append({"tokens" : torch.LongTensor(tokens),
                                    "mask" : torch.LongTensor(mask),
                                    "labels" : torch.LongTensor(labels),
                                    "token_id":torch.LongTensor(h_token_id_list)
                                    })
                    if b:
                        inputs.append({"tokens" : torch.LongTensor(tokens),
                                    "mask" : torch.LongTensor(mask),
                                    "labels" : torch.LongTensor(labels),
                                    "token_id":torch.LongTensor(t_token_id_list)
                                    })
            # 这里进行断点设置，查看原始的文本、实体以及对应的mask内容
            # import ipdb ;ipdb.set_trace()
            if self.config.get("external","is_test"):
                if len(inputs) > 1000:
                    break
        # import ipdb ;ipdb.set_trace()

        inputs_all = inputs
        return inputs_all
    
    def percentage_sample(self, data, percentage):
        import random
        """
        从给定数据中随机百分比采样
        :param data: 要采样的数据列表
        :param percentage: 采样的百分比（0到1之间的浮点数）
        :return: 采样结果列表
        """
        sample_size = int(len(data) * percentage)
        sampled_data = random.sample(data, sample_size)
        return sampled_data

    def process(self) :
        import pickle
        with open('code/all_input_MinReplace_full.pkl', 'rb') as file:
            all_input = pickle.load(file)

        # 数据百分比采样
        if self.config.get("external","get_sample"):
            all_input = self.percentage_sample(all_input, self.config.get("external","gs_rate"))

        return all_input



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    # config files and gpu
    parser.add_argument("--config", type=str, default='./mapping_networks/Wikipedia/Dropout25/default.config')
    parser.add_argument("--gpu", "-g", default = '0',type=str)
    parser.add_argument("--seed", type = int, default = 3407, choices=[42, 3407, 114514])
    parser.add_argument("--is_test", action='store_true')
    parser.add_argument("--is_tb", action='store_true')
    # Name of saved files and folds
    parser.add_argument("--name", default='test')
    parser.add_argument("--suffix", default='test')
    parser.add_argument("--title", type=str, default='keff')
    # Whether using modified model
    parser.add_argument("--run_mode", default='train')
    # 设置数据采样以及采样比
    parser.add_argument("--get_sample", "-gs", action='store_true')
    parser.add_argument("--gs_rate", type=float, default=1.0)
    # 设置映射网络的维度大小
    parser.add_argument("--network_dim", "-nd", type=int, default=148)
    args = parser.parse_args()
    
    configFilePath = args.config
    config = create_config(configFilePath)
    config.add_section("external")
    for item in list(vars(args)):
        config.set("external", item, vars(args)[item])
    # import ipdb ;ipdb.set_trace()

    # 设置使用的GPU
    # set_gpu(args.gpu)
    print(">> 设置使用的GPU……")
    best_gpu = get_most_free_memory_gpu()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(">> 设置随机种子……")
    seed_everything(args.seed)

    # 处理数据
    Formatter = MapTuing_Wikipedia_formatter(config)
    bert_embedding = Formatter.get_embedding()
    NeedMapper = Formatter.get_NeedMapper()
    print(">> 正在准备处理数据……")
    dataset = Formatter.process()
    Formatter = None
    print(">> 完成")

    # 定义映射器
    mapper = AffineMapper_KEFF(config.get("external", "network_dim")).to(device)
    # mapper.cuda()


    dataset = torch.utils.data.DataLoader(dataset = dataset, batch_size = 256, drop_last = False, shuffle = False, num_workers=4)
    output_path = f'./models_ckpt/{config.get("external", "title")}_{config.get("external", "network_dim")}'
    print(f'>> 设置模型的输出路径为: ./models_ckpt/{config.get("external", "title")}_{config.get("external", "network_dim")}')
    os.makedirs(output_path, exist_ok = True)
    
    
    tb_path = None
    if config.get("external","is_tb"):
        tb_path = './logs/'
        os.makedirs(tb_path, exist_ok = True)
        # import ipdb ;ipdb.set_trace()
    print(">> 转到训练环节……")
    train_new(config, mapper, dataset, bert_embedding, NeedMapper, output_path, 5, tb_path, config.get("external","name"), device)

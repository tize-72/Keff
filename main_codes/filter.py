from MapTuning_add_test import train_new
from formatter.basic_formatter import bert_formatter
import json
import torch
import os
from tqdm import tqdm
from tools.seed import seed_everything
from tools.gpu_tool import set_gpu
import argparse
from config_parser import create_config
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import trange
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import pickle
# import ipdb ;ipdb.set_trace()

import torch
from torch.utils.data import DataLoader, Subset

def create_subset_loader(dataset, batch_size=256, subset_ratio=0.1, shuffle=True, num_workers=0):
    """
    从原始 DataLoader 数据集中随机抽取指定比例的子集，并返回一个新的 DataLoader。

    :param dataset: 原始 DataLoader 或 Dataset 对象
    :param batch_size: 新 DataLoader 的 batch_size 大小
    :param subset_ratio: 子集占原始数据集的比例 (0-1)，默认值为 0.1 (即10%)
    :param shuffle: 是否对 DataLoader 进行打乱
    :param num_workers: DataLoader 使用的线程数
    :return: 包含随机子集的新 DataLoader 对象
    """
    # 确保 dataset 是 Dataset 对象（不是 DataLoader）
    if isinstance(dataset, DataLoader):
        dataset = dataset.dataset
    
    # 计算子集大小
    dataset_size = len(dataset)
    subset_size = int(dataset_size * subset_ratio)
    
    # 随机生成子集的索引
    random_indices = torch.randperm(dataset_size)[:subset_size].tolist()
    
    # 创建子集
    subset = Subset(dataset, random_indices)
    
    # 创建新的 DataLoader
    subset_loader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return subset_loader


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

class MapTuing_Wikipedia_formatter(bert_formatter) :
    def __init__(self, config) :
        super().__init__(config)
        self.config = config

    def get_format(self, mention, Qid) :
        original_mention = mention
        Qword = None
        if Qid not in self.keembedding.ent_map :
            Qid = None
        else :
            Qword = "hashdonothackme" + Qid.lower() + "hashdonothackme"
            token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" ".join(mention)))
        MASKS_original = ["[MASK]"] * len(self.tokenizer.tokenize(" ".join(mention)))
        MASKS = ["[MASK]"] * len(self.tokenizer.tokenize(" ".join(mention)))
        if Qid is None :
            pass
        else :
            mention = [Qword] + ["/"] + mention
            MASKS = [Qword] + ["/"] + MASKS_original
        return [" "] + mention + [" "], [" "] + MASKS + [" "], (Qid is not None), MASKS_original, [" "]+[Qword]+[" "], [" "]+original_mention+[" "]

    
    def template2sentence(self, text, h_word, t_word, pos1, pos2) :
        if pos1[0] < pos2[0] :
            text = text[: pos1[0]] + h_word + text[pos1[1] : pos2[0]] + t_word + text[pos2[1] :]
        else :
            text = text[: pos2[0]] + t_word + text[pos2[1] : pos1[0]] + h_word + text[pos1[1] :]
        # print(text)
        return " ".join(text)
    

    def get_sentence(self, inputs, text, pos1, pos2, tokens_text_list, label_h, label_t, token_h, token_t, token_id=None):
        label_text = self.template2sentence(text, 
                                            label_h, 
                                            label_t, 
                                            pos1, pos2)
        labels = self.tokenizer.tokenize(label_text)
        text_change = self.template2sentence(text, 
                                            token_h, 
                                            token_t,
                                            pos1, pos2)
        tokens = self.tokenizer.tokenize(text_change)

        # 设置标签
        for (i, _token) in enumerate(tokens) :
            if _token != "[MASK]" :
                assert(_token == labels[i])
                labels[i] = -100
            else :
                labels[i] = self.tokenizer.vocab[labels[i]]
        
        # 最终的数据放入inputs中
        tokens_text = ["[CLS]"] + tokens + ["[SEP]"]
        tokens_text_list.append(tokens_text)
        labels = [-100] + labels + [-100]
        assert(len(tokens_text) == len(labels))
        tokens_original = self.tokenizer.convert_tokens_to_ids(tokens_text)
        mask = [1] * len(tokens_original)
        padding = [0] * (self.max_length - len(tokens_original))
        labels += [-100] * (self.max_length - len(tokens_original))
        tokens = tokens_original + padding
        mask += padding
        assert(len(tokens) == len(labels))
        if token_id:
            inputs.append({"tokens" : torch.LongTensor(tokens).unsqueeze(0),
                    "mask" : torch.LongTensor(mask).unsqueeze(0),
                    "labels" : torch.LongTensor(labels).unsqueeze(0),
                    'token':tokens_original,
                    "token_id":torch.LongTensor(token_id).unsqueeze(0)})
        else:
            inputs.append({"tokens" : torch.LongTensor(tokens),
                        "mask" : torch.LongTensor(mask),
                        "labels" : torch.LongTensor(labels),
                        'token':tokens_original})

        return inputs, tokens_text_list


    # 取得实体相关的token中所有最小值的平均值
    def get_min_mean_value(self, sure_id,bert_embedding, item, zero_range, count_positive):
        min_mean = np.zeros(zero_range).astype("float32")
        indices_to_update_list = []
        for j in range(count_positive):
            if (sure_id + j) >= len(item):
                pass
            else:
                now_numpy_matrix = bert_embedding[item[sure_id + j]]
                sorted_id, _ = self.get_sort_list(list(np.abs(now_numpy_matrix)))
                indices_to_update = sorted_id[0:zero_range]
                indices_to_update_list.append(indices_to_update)
                min_mean += now_numpy_matrix[indices_to_update]
        min_mean = min_mean / count_positive
        return min_mean, indices_to_update_list


    # 取得所有token embedding值的最小位
    def get_min_mean_value_original(self, bert_embedding, item, zero_range):
        min_mean = np.zeros(zero_range).astype("float32")
        indices_to_update_list = []
        for j in range(len(item)):
            now_numpy_matrix = bert_embedding[item[j]]
            sorted_id, _ = self.get_sort_list(list(np.abs(now_numpy_matrix)))
            indices_to_update = sorted_id[0:zero_range]
            indices_to_update_list.append(indices_to_update)
            min_mean += now_numpy_matrix[indices_to_update]
        min_mean = min_mean / len(item)
        return min_mean


    def get_sort_list(self, my_list):
        # 使用enumerate获取元素和原始下标
        indexed_list = list(enumerate(my_list))

        # 根据元素的值进行排序
        sorted_list = sorted(indexed_list, key=lambda x: x[1])

        # 输出排序后的结果以及原始下标
        sorted_list_final = [item[1] for item in sorted_list]# print("排序后的列表：", [item[1] for item in sorted_list])
        sorted_id = [item[0] for item in sorted_list]# print("原始下标：", [item[0] for item in sorted_list])

        return sorted_id, sorted_list_final


    def Process(self, model, bert_embedding, NeedMapper, mapper, discriminator, name, output_path, is_tb, tb_path = './tb/') :
        mapper.eval()
        for mode in ("val", "test", "train") :
            if mode == "train":
                data_path = self.config.get("data", "data_path")
                with open(os.path.join(data_path, "wiki20m_{}.txt".format(mode)), "r", encoding = "utf-8") as fin :
                    data_use = [json.loads(line) for line in fin]
                data_list = []
                with trange(len(data_use)) as t:
                    import datetime
                    starttime = datetime.datetime.now()
                    t.ncols = 120
                    t.colour='green'
                    # import ipdb ;ipdb.set_trace();ccc
                    for iddddx in t:
                        instance = data_use[iddddx]
                        # 初始化inputs
                        inputs = []
                        tokens_text_list = []
                        loss_total = None
                        # 设置进度条左名称
                        t.set_description("DATA-Batch: {}".format(iddddx+1))
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
    
                        inputs, tokens_text_list= self.get_sentence(inputs, text, pos1, pos2, tokens_text_list, h_original_mention, t_original_mention, h_original_mention, t_mask)
                        inputs, tokens_text_list= self.get_sentence(inputs, text, pos1, pos2, tokens_text_list, h_original_mention, t_original_mention, h_original_mention, t_mask, h_token_id_list)
                        inputs, tokens_text_list= self.get_sentence(inputs, text, pos1, pos2, tokens_text_list, h_original_mention, t_original_mention, h_mask, t_original_mention, t_token_id_list)
                      
                        loss_original_h = model(inputs[0], bert_embedding, NeedMapper, mapper, True)
          
                        loss_mapper_h = model(inputs[1], bert_embedding, NeedMapper, mapper)
                        loss_original_t = model(inputs[0], bert_embedding, NeedMapper, mapper, True)
                        loss_mapper_t = model(inputs[2], bert_embedding, NeedMapper, mapper)
                        loss_cal = [loss_original_h, loss_mapper_h, loss_original_t, loss_mapper_t]

                        zero_range = self.config.get("external", "network_dim")
                        
                        data = inputs[0]
                        tokens, _, _ = data["tokens"], data["mask"], data["labels"]
                        device = ("cuda" if torch.cuda.is_available() else "cpu")

                        # 实体提及token最小位
                        data = inputs[1]
                        tokens, _, _, token_id = data["tokens"][0], data["mask"][0], data["labels"][0], data["token_id"][0]
                        vecs = torch.cat( [(torch.tensor(bert_embedding[id], device = device)).reshape((+1, -1))
                                                            for id in tokens], dim = 0)
                        count_positive = len(list(filter(lambda x: x > 0, token_id)))
                        if count_positive==0:
                            continue
                        for iddx, single_token in enumerate(tokens):
                            if single_token == token_id[0]:
                                sure_id = iddx
                        min_mean, indices_to_update_list = self.get_min_mean_value(sure_id, bert_embedding, tokens, zero_range, count_positive)
                        mapper_value = mapper(torch.tensor(min_mean, dtype=torch.float32, device=device))
                        vecs_list = []
                        for j in range(count_positive):
                            if (sure_id + j) >= len(tokens):
                                pass
                            else:
                                vecs[sure_id + j][indices_to_update_list[j]] =  mapper_value
                                vecs_list.append(vecs[sure_id + j])
                        mean_tensor = torch.mean(torch.stack(vecs_list), dim=0)

                        # 设置符号，进行二分类模型的训练
                        if loss_original_h.item() > loss_mapper_h.item():
                            label_h = torch.tensor(np.array([1], dtype=np.float32))
                        else:
                            label_h = torch.tensor(np.array([0], dtype=np.float32))
                        data_list.append((mean_tensor, label_h))


                        data = inputs[2]
                        tokens, _, _, token_id = data["tokens"][0], data["mask"][0], data["labels"][0], data["token_id"][0]
                        vecs = torch.cat( [(torch.tensor(bert_embedding[id], device = device)).reshape((+1, -1))
                                                            for id in tokens], dim = 0)
                        count_positive = len(list(filter(lambda x: x > 0, token_id)))
                        for iddx, single_token in enumerate(tokens):
                            if single_token == token_id[0]:
                                sure_id = iddx
                        # 实体提及token最小位
                        min_mean, indices_to_update_list = self.get_min_mean_value(sure_id, bert_embedding, tokens, zero_range, count_positive)
                        mapper_value = mapper(torch.tensor(min_mean, dtype=torch.float32, device=device))
                        vecs_list = []
                        if count_positive==0:
                            continue
                        for j in range(count_positive):
                            if (sure_id + j) >= len(tokens):
                                pass
                            else:
                                vecs[sure_id + j][indices_to_update_list[j]] =  mapper_value
                                vecs_list.append(vecs[sure_id + j])
                        mean_tensor = torch.mean(torch.stack(vecs_list), dim=0)
                        # 设置符号，进行二分类模型的训练
                        if loss_original_t.item() > loss_mapper_t.item():
                            label_t = torch.tensor(np.array([1], dtype=np.float32))
                        else:
                            label_t = torch.tensor(np.array([0], dtype=np.float32))
                        data_list.append((mapper_value, label_t))
                        
                        endtime = datetime.datetime.now()
                        delta_t = ((endtime - starttime).seconds)/60
                        t.set_postfix(delta_time = round(delta_t,3))
                        if len(data_list) >  140000: break
                        
                    t.close()
                dataset_set = MyDataset(data_list)
                
                dataset = DataLoader(dataset=dataset_set, batch_size=256, shuffle=True, num_workers=0, drop_last=False)

                return dataset



# 设置分辨器网络结构
class Discriminator(torch.nn.Module) :
    # 分辨器网络返回的是一个长度为2的预测值，用sigmoid激活函数进行激活
    def __init__(self, input_dim) :
        super().__init__()

        self.f1 = torch.nn.Linear(input_dim, 1)

    def abandon_grad(self) :
        self.model.requires_grad_ = False
    def forward(self, x) :
        x = self.f1(x)

        
        x = F.sigmoid(x)
        return x
    
    def save(self, path) :
        torch.save(self.state_dict(), path)
    def load(self, path) :
        if path != "None" :
            self.load_state_dict(torch.load(path, map_location = torch.device("cpu")))
        if torch.cuda.is_available() :
            self.model = self.model.cuda()
        self.eval()
    def trained_parameters(self) :
        return self.parameters()
    

if __name__ == "__main__" :
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    # 主程序入口
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default='../mapping_networks/Wikipedia/Dropout25/default.config')
    parser.add_argument("--gpu", "-g", default = None)
    parser.add_argument("--drop_rate", "-dr", type=float, default = 0.1)
    parser.add_argument("--seed", type = int, default = 3407)
    parser.add_argument("--is_test", action='store_true')
    parser.add_argument("--is_tb", action='store_true')
    parser.add_argument("--name", default='test')
    parser.add_argument("--suffix", default='test')
    parser.add_argument("--run_mode", default='train')
    parser.add_argument("--is_merge", action='store_true')
    parser.add_argument("--network_dim", "-nd", type=int, default=149)
    parser.add_argument("--round", "-r", type=int, default=2)


    args = parser.parse_args()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    configFilePath = args.config
    config = create_config(configFilePath)
    config.add_section("external")
    for item in list(vars(args)):
        config.set("external", item, vars(args)[item])

    # 设置gpu与随机种子等
    set_gpu(args.gpu)
    seed_everything(args.seed)

    # set formatter
    Formatter = MapTuing_Wikipedia_formatter(config)
    bert_embedding = Formatter.get_embedding()
    NeedMapper = Formatter.get_NeedMapper()


    discriminator = Discriminator(768)
    discriminator = discriminator.cuda()

    # 设置映射网络
    from mapper import AffineMapper, MergeMapNetwork, MergeMapNetwork_768, MergeMapNetwork_add_drop, AffineMapper_KEFF
    mapper = AffineMapper_KEFF(config.get("external","network_dim"))
    # mapper_path = '../mapping_networks/Wikipedia/{}/Dropout25/Affine_{}.bin'.format(config.get("external","suffix"), config.get("external","round"))
    mapper_path = f'../models_ckpt/keff_{config.get("external", "network_dim")}/Affine_{config.get("external", "round")}.bin'
    mapper.load(mapper_path)
    mapper.cuda()
    from MapTuning_add_test import MapTuning_MLM_add

    # 设置bert网络用来进行模型的loss输出判断
    model = MapTuning_MLM_add(config, device) # 设置模型网络
    model = model.cuda()


    output_path = '../models_ckpt/filter/'
    dataset = Formatter.Process(model, bert_embedding, NeedMapper, mapper, discriminator, config.get("external","name"), output_path, config.get("external","is_tb"))
    # 保存变量
    with open('filter_dataset.pkl', 'wb') as aaa:
        pickle.dump(dataset, aaa)

    dataset = create_subset_loader(dataset, batch_size=128, subset_ratio=0.1, shuffle=True, num_workers=0)

    #设置tensorboard的路径
    if config.get("external", "is_tb"):
        tb_path = './dis_tb/'
        os.makedirs(tb_path, exist_ok = True)
        from torch.utils.tensorboard import SummaryWriter # type: ignore
        from datetime import datetime
        tensorboardLogPath = os.path.join(
                        tb_path, config.get("external", "name")
                    )
        os.makedirs(tensorboardLogPath, exist_ok = True)
        modelWriter = SummaryWriter(log_dir=tensorboardLogPath)

    optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    epochs = 1000
    import datetime
    epoch_starttime = datetime.datetime.now()
    with trange(epochs) as t:
        
        for epoch in t:
            trainLoss = 0
            t.set_description("Epoch:{}".format(epoch+1))
            train_iter = iter(dataset)#--|迭代器生成
            for batch_idx in range(len(dataset)):
                emb, label = next(train_iter)
                emb, label = emb.cuda(), label.cuda()
                discriminator.train()
                optimizer.zero_grad()
                pred = discriminator(emb)
                loss = criterion(pred, label.float())
                loss.backward(retain_graph=True)
                # import ipdb ;ipdb.set_trace()
                trainLoss += loss.item()
                endtime = datetime.datetime.now()
                optimizer.step()
                # import ipdb ;ipdb.set_trace()
            tensorLossWriteDict = {'Loss': trainLoss / len(dataset)
                                }
            tb_name = config.get("external", "suffix")
            if config.get("external", "is_tb"):
                        modelWriter.add_scalars('{}'.format(tb_name+'Loss'), tensorLossWriteDict, epoch)
            epoch_endtime = datetime.datetime.now()
            delta_t = ((epoch_endtime - epoch_starttime).seconds)/60
            t.set_postfix(loss=round(float(trainLoss / len(dataset)), 3), delta_time = round(delta_t,3))
        model_to_save = discriminator.module if hasattr(discriminator, "module") else discriminator
        model_to_save.save(os.path.join(output_path, f'{config.get("external", "network_dim")}_{config.get("external", "round")}_filter{epoch+1}.bin'))
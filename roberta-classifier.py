# coding=utf-8
#
# Copyright (c) 2022
# Frederick Yuanchun Wang
# frederickwang99@gmail.com
#
# All rights reserved.
#     
#  ___      ___       __       ___
#   \_ \_   \_ \_   _/  \_   _/ _/
#     \  \    \_ \_/ _/\_ \_/ _/
#     \all \    \___/    \___/
#    /rights\     \_ \__/ _/
#   /reserved       \_  _/    ✿
#  | WangYC /      _/__/_ _ _`|'
#  \  99   /     _/ _ _ _ _ _/
#   \  ©  /    _/ _/
#     \_  \  _/ _/_ _ _ _ _
#       \_\_/_ _ _ _ _ _ _/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['http_proxy'] = '127.0.0.1:34567'
os.environ['HF_HOME'] = '/zhangpai25/wyc/lewis/lewis_wyc/hf_data/'

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
import torch.utils.data as Data
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime


# for n, m in roberta.named_modules():
#     print(n)
# okenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base", cache_dir='/zhangpai25/wyc/lewis/lewis_wyc/hf_data/')
# roberta = BertModel.from_pretrained("clue/roberta_chinese_base", cache_dir='/zhangpai25/wyc/lewis/lewis_wyc/hf_data/')


# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = roberta(**inputs) 
# output中的内容：
    # pooler
    # pooler.dense
    # pooler.activation
    # last_hidden_state
    # pooler_output

# print(outputs.last_hidden_state.shape)

parser = argparse.ArgumentParser() # 初始化

parser.add_argument('--batch_size', type=int, default=64,
                       help='size of one batch') 
parser.add_argument('--epoch', type=int, default=100,
                       help='epoch')
parser.add_argument('--learning_rate', type=float, default=1e-6,
                       help='learning_rate')
parser.add_argument('--val_per_ite', type=int, default=50,
                       help='validation per how many iterations')
parser.add_argument('--model_save', default='/zhangpai25/wyc/lewis/lewis_wyc/saved_ckpts',
                       help='path to ckpt save dir')
parser.add_argument('--device', default='cpu',
                       help='the device you want to use to train')
args, others_list = parser.parse_known_args() # 解析已知参数

def load_data(arg_mode):
    # """用来生成训练、测试数据"""
    # train_df = pd.read_csv("bert_example.csv", header=None)
    # sentences = train_df[0].values
    # targets = train_df[1].values
    # train_inputs, test_inputs, train_targets, test_targets = train_test_split(sentences, targets)
    if arg_mode == 'train':
        typed_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm_large/domain-1-typed/train.txt'
        untyped_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm_large/domain-2-untyped/train.txt'
    elif arg_mode == 'val':
        typed_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm_large/domain-1-typed/valid.txt'
        untyped_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm_large/domain-2-untyped/valid.txt'

    train_inputs = []
    train_targets = []
    # with open(data_dir, 'r') as file_src:
    #     train_src = json.load(file_src)
    # file_src.close()
    file = open(typed_data_dir)
    typed_sentences = file.read()
    typed_sentences_list = typed_sentences.split('\n')
    file.close()

    file = open(untyped_data_dir)
    untyped_sentences = file.read()
    untyped_sentences_list = untyped_sentences.split('\n')
    file.close()
    import random
        
    typed_iterator = 0
    untyped_iterator = 0
    for iterator in range(2 * len(typed_sentences_list)):
        decision = random.randint(0, 1)
        # print(decision)
        if decision == 0:
            try:
                train_inputs.append(typed_sentences_list[typed_iterator])
                train_targets.append(0)
                typed_iterator += 1
            except:
                continue
        else:
            try:
                train_inputs.append(untyped_sentences_list[untyped_iterator])
                train_targets.append(1)
                untyped_iterator += 1
            except:
                continue
                
    return train_inputs, train_targets

class RoBERTa_classifier(nn.Module):
    def __init__(self) :
        super(RoBERTa_classifier, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base", cache_dir='/zhangpai25/wyc/lewis/lewis_wyc/hf_data/')
        self.roberta = RobertaModel.from_pretrained("clue/roberta_chinese_base", cache_dir='/zhangpai25/wyc/lewis/lewis_wyc/hf_data/')


        self.classifier_1 = nn.Linear(768, 1024)
        self.classifier_2 = nn.Linear(1024, 2)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, batch_sentences):
        sentence_tokenized = self.tokenizer(batch_sentences,
                                            truncation=True,
                                            padding=True,  
                                            max_length=30,  
                                            add_special_tokens=True)  
        input_ids = torch.tensor(sentence_tokenized['input_ids']) 
        attention_mask = torch.tensor(sentence_tokenized['attention_mask']) 
        # input_ids = input_ids.to(args.device)
        # attention_mask = attention_mask.to(args.device)
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        # print(roberta_output.last_hidden_state.shape)
        logits = roberta_output.last_hidden_state[:, 0, :] 
        # for each in roberta_output:
        #     print(each)
        # exit()
        # logits = self.use_bert_classify(bert_cls_hidden_state)
        # logits = roberta_output.last_hidden_state[0]
        result = self.sigmoid(self.classifier_2(self.sigmoid(self.classifier_1(logits))))

        prediction = self.softmax(result)

        final_result = torch.argmax(prediction, dim=-1).float()

        return prediction, final_result

# feature接收
class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out

# register
def get_feas_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if n == "encoder.layer.10.attention.self.query":
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks

def train(args):
    print("start loading data...")
    train_inputs, train_targets = load_data('train')

    print("data loaded! ")
    # epochs = args.epoch
    # batch_size = args.batch_size
    # data_dir = args.data_dir
    epochs = args.epoch
    batch_size = args.batch_size

    train_sentence_loader = Data.DataLoader(
        dataset=train_inputs,
        batch_size=batch_size,  # 每块的大小
    )
    train_label_loader = Data.DataLoader(
        dataset=train_targets,
        batch_size=batch_size,
    )
    
    roberta_classifier_model = RoBERTa_classifier()
    # roberta_classifier_model = RoBERTa_classifier.to(args.device)
    # print(roberta_classifier_model.state_dict()['roberta.encoder.layer.11.output.LayerNorm.bias'])
    # for name, pa in roberta_classifier_model.named_parameters():
    #     print(name)
    # exit()

    fea_hooks = get_feas_by_hook(roberta_classifier_model) # 调用函数，完成注册即可

    print('The number of hooks is:', len(fea_hooks))

    # for name, each in bert_classifier_model.named_modules():
    #     if name == "bert.encoder.layer.11":
    #         print(each.named_children())

    optimizer = torch.optim.Adam(roberta_classifier_model.parameters(), lr=args.learning_rate)

    # criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = torch.nn.MSELoss(reduction='mean')
    # criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    # criterion = F.binary_cross_entropy()
    # criterion = criterion.to(args.device)

    print("training start...")
    roberta_classifier_model.train()

    iteration = 0
    best_eval = 0
    es_count = 0

    for epoch in range(epochs): # 开始训练
        print('...this is epoch : {}...'.format(epoch))
        loss_list = []        
        for sentences, labels in zip(train_sentence_loader, train_label_loader):
            # print('sentences are : {} and labels are: {}'.format(sentences, labels))
            # exit()
            # labels = labels.to(args.device)
            # print (labels)
            # break
            result, cla_result = roberta_classifier_model(sentences)
            
            acc = 0
            for iterator in range(len(result)):
                if cla_result[iterator] == labels[iterator]:
                    acc += 1
                else:
                    continue
            acc = acc / len(result)

            binary_labels = []
            for each in labels:
                if each == 0:
                    binary_labels.append([1., 0.])
                else:
                    binary_labels.append([0., 1.])
            binary_labels = np.array(binary_labels,dtype=float)
            binary_labels = torch.tensor(binary_labels)

            # print('The shape of the last bert layer feature is:', fea_hooks[0].fea[0].shape)
            # print('result', cla_result)
            # print('labels', labels)
            
            # print(binary_labels)
            # exit()
            # x=Variable(x,requires_grad=True)
            loss = F.binary_cross_entropy_with_logits(result, binary_labels)
            # loss.retain_grad()
            # loss.requires_grad_(True)
            # loss.requ

            writer.add_scalar("train_loss", loss, iteration)
            writer.add_scalar("train_acc", acc, iteration)

            optimizer.zero_grad()
            loss.backward()
            print('this is iteration : {} and loss : {}'.format(iteration, loss))
            # break
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            iteration += 1
            if iteration and iteration % args.val_per_ite == 0:
                torch.save(roberta_classifier_model.state_dict(), args.model_save + '/saved.pkl')
                roberta_classifier_model.eval()
                valid_loss, acc = valid(args)
                writer.add_scalar("valid_loss", valid_loss, iteration)
                writer.add_scalar("valid_acc", acc, iteration)
                if acc >= best_eval:
                    best_eval = acc
                    es_count = 0
                else:
                    es_count += 1
                
                if es_count == 30:
                    print('early_stopping!!!')
                    break
    torch.save(roberta_classifier_model.state_dict(), args.model_save + '/final.pkl')

def valid(args):
    print("validation start...")
    print("start loading data...")
    train_inputs, train_targets = load_data('val')

    print("data loaded! ")
    # epochs = args.epoch
    # batch_size = args.batch_size
    # data_dir = args.data_dir
    epochs = args.epoch
    batch_size = args.batch_size

    train_sentence_loader = Data.DataLoader(
        dataset=train_inputs,
        batch_size=batch_size,  # 每块的大小
    )
    train_label_loader = Data.DataLoader(
        dataset=train_targets,
        batch_size=batch_size,
    )
    
    roberta_classifier_model = RoBERTa_classifier()
    roberta_classifier_model.load_state_dict(torch.load(args.model_save + '/saved.pkl'))
    # roberta_classifier_model = RoBERTa_classifier.to(args.device)
    # print(roberta_classifier_model.state_dict()['roberta.encoder.layer.11.output.LayerNorm.bias'])
    # for name, pa in roberta_classifier_model.named_parameters():
    #     print(name)
    # exit()

    # fea_hooks = get_feas_by_hook(roberta_classifier_model) # 调用函数，完成注册即可

    # print('The number of hooks is:', len(fea_hooks))

    # for name, each in bert_classifier_model.named_modules():
    #     if name == "bert.encoder.layer.11":
    #         print(each.named_children())

    # optimizer = torch.optim.Adam(roberta_classifier_model.parameters(), lr=1e-5)

    # criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = torch.nn.MSELoss(reduction='mean')
    # criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    # criterion = F.binary_cross_entropy()
    # criterion = criterion.to(args.device)
    roberta_classifier_model.eval()

    iteration = 0

    valid_loss = 0
    valid_acc = 0
     
    for sentences, labels in zip(train_sentence_loader, train_label_loader):
        # print('sentences are : {} and labels are: {}'.format(sentences, labels))
        # exit()
        # labels = labels.to(args.device)
        # print (labels)
        # break
        result, cla_result = roberta_classifier_model(sentences)

        binary_labels = []
        for each in labels:
            if each == 0:
                binary_labels.append([1., 0.])
            else:
                binary_labels.append([0., 1.])

        # print('The shape of the last bert layer feature is:', fea_hooks[0].fea[0].shape)
        # print('result', result)
        binary_labels = np.array(binary_labels,dtype=float)
        binary_labels = torch.tensor(binary_labels)
        # print('labels', binary_labels)
        # print(binary_labels)
        # exit()
        # x=Variable(x,requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(result, binary_labels)
        valid_loss += loss.item()
        acc = 0
        for iterator in range(len(result)):
            if cla_result[iterator] == labels[iterator]:
                acc += 1
            else:
                continue
        acc = acc / len(result)
        valid_acc += acc
        # loss.retain_grad()
        # loss.requires_grad_(True)
        # loss.requ
        iteration += 1
    
    valid_loss = valid_loss / len(sentences)
    valid_acc = valid_acc / len(sentences)
    print('validation finished, loss:{}, acc:{}'.format(valid_loss, valid_acc))
    return valid_loss, valid_acc


if __name__ == '__main__':
    time = datetime.datetime.now()
    time_str = str(time.month) + '-' + str(time.day) + '-' + str(time.hour) + '-' + str(time.minute)
    writer = SummaryWriter('./logs/' + time_str)
    os.system("mkdir " + args.model_save + '/' + time_str)
    args.model_save = args.model_save + '/' + time_str
    train(args)
    # eval(args)
    # test()
    print('done :-)')
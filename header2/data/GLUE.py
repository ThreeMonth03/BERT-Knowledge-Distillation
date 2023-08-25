from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import Dataset, Subset
import torch.nn as nn
import os
import numpy as np
# from embed import Embeddings
# flag = ['train, validation, test']

os.environ["TOKENIZERS_PARALLELISM"] = "false"
class GLUEDataset(Dataset):
    def __init__(
        self,
        task: str,
        max_length: int = 256,
        ckpt: str = 'roberta-base',
        flag: str = 'train',
        part: float = 1.0,
        mask_invert: bool = False,
        rand_mask_part: float = 0.0,
    ) -> None:
        """
        tokenizer candidate[
            nghuyong/ernie-2.0-en,
            bert-base-uncased'
        ]
        """
        
        self.mask_invert = mask_invert
        self.max_length = max_length
        # self.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast = True)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        
        # set downstream task
        task_to_keys = {
            "sst2": ("sentence", None),        #67349      二分類 正負情感     => two central performances 1   so pat it makes your teeth hurt 0
            "mrpc": ("sentence1", "sentence2"), #3668      二分類和F1值  語意是否一樣
            "qnli": ("question", "sentence"),  #104743     判断问题（question）和句子（sentence，维基百科段落中的一句）是否蕴含，蕴含和不蕴含，二分类。

            "mnli": ("premise", "hypothesis"), #392702     三分類 句子对，一个前提，一个是假设。前提和假设的关系有三种情况：蕴含（entailment），矛盾（contradiction），中立（neutral）。句子对三分类问题。
            "qqp": ("question1", "question2"), #363846     二分類和F1值 句子是否等效 => How can I improve my communication and verbal skills? What should we do to improve communication skills?
            
            
            "cola": ("sentence", None),        #8551       二分類(Matthews correlation coefficient。) 合乎語法與否 => She is proud.1   Mary sent.0  單句 
            "rte": ("sentence1", "sentence2"), #2490       判断句子对是否蕴含，句子1和句子2是否互为蕴含，二分类任务。 
            "stsb": ("sentence1", "sentence2"),#5749       回归任务，预测为1-5之间的相似性得分的浮点数也可以當作五分類 => A plane is taking off. An air plane is taking off. 5.000
            "wnli": ("sentence1", "sentence2"),#635        判断句子对是否相关，蕴含和不蕴含，二分类任务。
        }
        task = task.lower()
        if task not in task_to_keys.keys():
            raise ValueError("I don't know what you mean.  Please reset task flag!")
        GlueDataset = load_dataset('glue', task)[flag]
        # print(GlueDataset)
        self.sentence1_key , self.sentence2_key = task_to_keys[task]
        
        # split data
        num = int(len(GlueDataset) * part)
        self.dataset = Subset(GlueDataset, range(num))
        if flag=='train':
            self.rand_mask_part = 1.0 - rand_mask_part
        else:
            self.rand_mask_part = 1.0
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        examples = self.dataset[idx]
        if self.sentence2_key:
            data = self.tokenizer(
                examples[self.sentence1_key],
                examples[self.sentence2_key],
                truncation = True,
                padding = "max_length",
                max_length = self.max_length
            )
        else:
            data = self.tokenizer(
                examples[self.sentence1_key],
                truncation = True,
                padding = "max_length",
                max_length = self.max_length
            )
        
        x = torch.IntTensor(data['input_ids'])
        x_mask = torch.BoolTensor(data['attention_mask'])
        rand_mask = torch.rand(torch.sum(x_mask == True)) < self.rand_mask_part
        temp = torch.rand(self.max_length - rand_mask.size(0)) < 0
        rand_mask = torch.cat((rand_mask, temp), dim=-1)
        x_mask = (x_mask & rand_mask)
        if self.mask_invert:
            x_mask = (x_mask == False)
        y = examples['label']
        
        # process sentence id
        sep_id = (torch.IntTensor(data['input_ids']) == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        setence_id = torch.zeros(self.max_length).to(int)
        if sep_id.shape[0] == 3:
            setence_id[sep_id[1]:sep_id[2]] = 1
            # setence_id[sep_id[2]:] = 2
        elif sep_id.shape[0] == 2:
            setence_id[sep_id[1]:] = 1

        return x, x_mask, setence_id, y


if __name__ == '__main__':

    data_set = GLUEDataset(flag='train', task='mnli', rand_mask_part=0.2)
    # print(data_set[0])
    for i in data_set:
        print(i[3])

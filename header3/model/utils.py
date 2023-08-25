import os
import random
import numpy as np
from typing import Any
import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_data(dataset: Any, batch_size: int, shuffle: bool, sampler=None, num_workers: int = -1) -> DataLoader:
    if num_workers == -1:
        num_workers = len(os.sched_getaffinity(0))

    if sampler:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True
        )

    return data_loader


class ProgressBar:
    last_length = 0

    @staticmethod
    def show(prefix: str, postfix: str, current: int, total: int, newline: bool = False) -> None:
        progress = (current + 1) / total
        if current == total:
            progress = 1

        current_progress = progress * 100
        progress_bar = '=' * int(progress * 20)

        message = ''

        if len(prefix) > 0:
            message += f'{prefix}, [{progress_bar:<20}]'

            if not newline:
                message += f' {current_progress:6.2f}%'

        if len(postfix) > 0:
            message += f', {postfix}'

        print(f'\r{" " * ProgressBar.last_length}', end='')
        print(f'\r{message}', end='')

        if newline:
            print()
            ProgressBar.last_length = 0
        else:
            ProgressBar.last_length = len(message) + 1

            
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_accuracy_min = 0.0
        self.delta = delta
        
        self.path = path
        directory = '/'.join(path.split('/')[: -1])
        os.makedirs(directory, exist_ok=True)
        
        self.trace_func = trace_func

    def __call__(self, val_accuracy, model, type):
        score = val_accuracy

        if(type == 'accuracy'):
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_accuracy, model, type=type)
            elif score < self.best_score + self.delta:
                self.counter += 1

                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
                if self.counter >= self.patience:
                    self.trace_func('Early stop!!!!!\n')
                    return False
            else:
                self.best_score = score
                self.save_checkpoint(val_accuracy, model, type=type)
                self.counter = 0
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_accuracy, model, type=type)
            elif score > self.best_score:
                self.counter += 1

                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
                if self.counter >= self.patience:
                    self.trace_func('Early stop!!!!!\n')
                    return False
            else:
                self.best_score = score
                self.save_checkpoint(val_accuracy, model, type=type)
            
        return True

    def save_checkpoint(self, val_accuracy, model, type):
        '''Saves model when validation accuracy decrease.'''
        if self.verbose:
            if(type == 'accuracy'):
                self.trace_func(f'Validation accuracy increased ({self.val_accuracy_min:.6f} --> {val_accuracy:.6f}).  Saving model ...\n')
            else:
                self.trace_func(f'loss decreased ({self.val_accuracy_min:.6f} --> {val_accuracy:.6f}).  Saving model ...\n')
        torch.save(model.state_dict(), self.path)
        self.val_accuracy_min = val_accuracy
        
def layerSelect(model, dataloader, t_layers, s_layers, device):
    model = model.to(device)
    model.eval()
    loss = torch.zeros(t_layers).to(device)
    for x, x_mask, sentence_id, y in dataloader:
        x = x.to(device)
        x_mask = x_mask.to(device)
        sentence_id = sentence_id.to(device)
        y = y.to(device)
            
        with torch.no_grad():
            t_outputs = model(x, x_mask, sentence_id)
            t_attention = t_outputs['attentions']
            t_average_attention_map = torch.zeros(t_attention[0].shape).to(device)
            
            for t_att in t_attention:
                t_average_attention_map = torch.add(t_average_attention_map, t_att)
            t_average_attention_map = torch.div(t_average_attention_map, len(t_attention))
            
            
            for i in range(len(t_attention)):
                loss[i] += F.mse_loss(t_attention[i], t_average_attention_map, reduction='mean')
            
    
    loss = loss.cpu().tolist()
    print(loss)
    priority_idx = sorted(range(len(loss)), key=lambda k: loss[k])
    priority_idx.reverse()
    priority_idx = priority_idx[:s_layers]
    priority_idx = sorted(priority_idx)
    return priority_idx

### new selection layer
# def layerSelect(model, dataloader, t_layers, s_layers, device):

#     model = model.to(device)
#     model.eval()
    
#     init = 0
    
#     for x, x_mask, sentence_id, y in dataloader:
#         x = x.to(device)
#         x_mask = x_mask.to(device)
#         sentence_id = sentence_id.to(device)
#         y = y.to(device)
#         with torch.no_grad():
#             t_outputs = model(x, x_mask, sentence_id)
#             t_attention = t_outputs['attentions']

        
#         if init == 0:
#             t_attention = list(t_attention)
#             for i in range(len(t_attention)):
#                 t_attention[i] = torch.sum(t_attention[i], 0)
#             total_attention = t_attention.copy()
#             init = 1
#         else:
#             for i in range(len(total_attention)):
#                 total_attention[i] = torch.add(total_attention[i], torch.sum(t_attention[i], 0))
#         break
    
    
#     center_attention = torch.zeros(total_attention[0].shape).to(total_attention[0].device)
#     for attention in total_attention:
#         center_attention = torch.add(center_attention, attention)
#     center_attention = torch.div(center_attention, len(total_attention))

    
#     not_select_layers = list(range(t_layers))
#     select_layer = []
    
#     for i in range(s_layers):
#         max_distance = 0
#         temp_select_layer = -1
#         if len(select_layer) == 0:
#             select_center = center_attention
#         else:
#             select_center = torch.zeros(total_attention[0].shape).to(total_attention[0].device)
#             for layer in select_layer:
#                 select_center = torch.add(select_center, total_attention[layer])
#             select_center = torch.div(select_center, len(select_layer))
        
#         for layer in not_select_layers:
#             temp_distance = F.mse_loss(select_center, total_attention[layer], reduction='mean')
#             if temp_distance > max_distance:
#                 max_distance = temp_distance
#                 temp_select_layer = layer
#         select_layer.append(temp_select_layer)
#         not_select_layers.remove(temp_select_layer)
    
#     return sorted(select_layer)






if __name__ == '__main__':
    pass
from typing import Any, Callable, Optional, Sequence, Dict, Tuple, Union, List
from sklearn.metrics import r2_score
import torch
import torch.optim as optim
import torch.cuda as cuda
from torch.utils.data import DataLoader
import torch.nn as nn
from .utils import ProgressBar
import torch.nn.functional as F
from .loss import Distillation

from sklearn.metrics import f1_score

# metadistill
from copy import deepcopy as cp
from collections import OrderedDict

import evaluate

class _History:
    def __init__(self, metrics: Sequence[str] = ['loss', 'accuracy'], additional_keys: Sequence[str] = []) -> None:
        self.metrics = metrics
        self.additional_keys = additional_keys

        self._history = {
            'count': [],
            'loss': [],
            'correct': [],
            'accuracy': [],
            'f1score': []
        }

        for key in self.additional_keys:
            self._history[key] = []

    def __str__(self) -> str:
        results = []

        for metric in self.metrics:
            results.append(f'{metric}: {self._history[metric][-1]:.6f}')

        return ', '.join(results)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, float]]:
        results = {}

        for metric in self.metrics:
            results[metric] = self._history[metric][idx]

        return results

    def reset(self) -> None:
        for key in self._history.keys():
            self._history[key].clear()

    def log(self, key: str, value: Any) -> None:
        self._history[key].append(value)

        if len(self._history['count']) == len(self._history['correct']) and len(self._history['count']) > len(self._history['accuracy']):
            self._history['accuracy'].append(self._history['correct'][-1] / self._history['count'][-1])

    def summary(self) -> None:
        _count = sum(self._history['count'])
        if _count == 0:
            _count = 1

        _loss = sum(self._history['loss']) / len(self._history['loss'])
        _correct = sum(self._history['correct'])
        _accuracy = _correct / _count

        self._history['count'].append(_count)
        self._history['loss'].append(_loss)
        self._history['correct'].append(_correct)
        self._history['accuracy'].append(_accuracy)

        for key in self.additional_keys:
            _value = sum(self._history[key]) / len(self._history[key])
            self._history[key].append(_value)


class _BaseRunner:
    def __init__(self, device='cuda') -> None:
        self.device = device if cuda.is_available() else 'cpu'

    @property
    def weights(self) -> None:
        raise NotImplementedError('weights not implemented')


class GLUERunner(_BaseRunner):
    def __init__(
        self,
        teacher,
        student,
        optimizer: optim.Optimizer,
        criterion,
        model_ckpt: Optional[Callable] = None,
        device: str = 'cuda:0',
        train_flag: int = 1,

    ) -> None:
        super().__init__(device=device)

        self.history = _History(metrics=['loss', 'accuracy','correct','f1score'])
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.model_ckpt = model_ckpt
        self.train_flag = train_flag

    
    def _step(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        sentence_id: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        
        x = x.to(self.device)
        x_mask = x_mask.to(self.device)
        sentence_id = sentence_id.to(self.device)
        y = y.to(self.device)
        #####

        # write knowledge distillation code
        # hint intermediate = outputs['hidden_states']
        # calculate loss

        #####
        ##student        
        s_outputs = self.student(x, x_mask, sentence_id)
        s_output = s_outputs['logits']
        #s_hidden = s_outputs['hidden_states']
        ##teacher
        #t_outputs = self.teacher(x, x_mask, sentence_id)
        #t_output = t_outputs['logits']
        #t_hidden = t_outputs['hidden_states']       
        ##Distillation model
        #D_model = Distillation(trainingConfig = "")
        
        s_y_hat = torch.argmax(s_output, dim=-1)
        running_loss = self.criterion(s_output, y.squeeze()) #+ D_model.soft_label_loss(t_output, s_output,'kd') + D_model.hidden_state_loss(t_hidden, s_hidden)

        nn.utils.clip_grad_value_(self.student.parameters(), clip_value=1.0)
        
        

        self.history.log('count', y.shape[0])
        self.history.log('loss', running_loss)
        self.history.log('correct', int(torch.sum(s_y_hat == y.squeeze()).item()))
        self.history.log('f1score', f1_score(y.cpu(), s_y_hat.cpu()))
        return running_loss
    
    
    def train(
            self,
            epochs: int, 
            train_loader: DataLoader, 
            valid_loader: Optional[DataLoader] = None, 
            scheduler: Any = None,
            save_type: str = 'accuracy'
        ) -> None:
        epoch_length = len(str(epochs))
        
        
        
        for epoch in range(epochs):
           
            self.teacher.eval()
            self.student.train()
            for i, (x, x_mask, sentence_id, y) in enumerate(train_loader):
                
                running_loss = self._step(x, x_mask, sentence_id, y)

                self.optimizer.zero_grad()
                running_loss.backward()  
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 0.5)
                self.optimizer.step()

                prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                postfix = str(self.history)
                ProgressBar.show(prefix, postfix, i, len(train_loader))

            self.history.summary()

            prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, len(train_loader), len(train_loader), newline=True)
            self.history.reset()
            
            if valid_loader:
                flag = self.val(valid_loader, type = save_type)
                if not flag:
                    break

            if scheduler:
                scheduler.step()
            
        
    @torch.no_grad()
    def val(self, test_loader: DataLoader, type: str='accruacy') -> None:
        self.student.eval()
        flag = True
        
        for i, (x, x_mask, sentence_id, y) in enumerate(test_loader):
            running_loss = self._step(x, x_mask, sentence_id, y)
            prefix = 'Val'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, i, len(test_loader))
        self.history.summary()

        prefix = 'Val'
        postfix = str(self.history)
        ProgressBar.show(prefix, postfix, len(test_loader), len(test_loader), newline=True)

        if self.model_ckpt is not None:
            if type == 'accuracy':
                flag = self.model_ckpt(self.history[-1]['accuracy'], self.student, type=type)
            else:
                flag = self.model_ckpt(self.history[-1]['loss'], self.student, type=type)
        self.history.reset()
        return flag


    @torch.no_grad()
    def test(self, test_loader: DataLoader, type: str='accruacy') -> None:
        self.student.eval()
        self.pre = torch.empty(0).to(self.device)
        self.ref = torch.empty(0).to(self.device)
        for i, (x, x_mask, sentence_id, y) in enumerate(test_loader):
            running_loss = self._step(x, x_mask, sentence_id, y)
            prefix = 'Test'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        self.history.summary()

        prefix = 'Test'
        postfix = str(self.history)
        ProgressBar.show(prefix, postfix, len(test_loader), len(test_loader), newline=True)
        self.history.reset()

    @property
    @torch.no_grad()
    def weights(self):
        return {'net': self.student}


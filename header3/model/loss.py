import torch
import torch.nn as nn
import torch.nn.functional as F


class Distillation(nn.Module):
    def __init__(
        self,
        trainingConfig
    ):
        super(Distillation, self).__init__()
        #self.trainingConfig = trainingConfig
        
        #self.alpha_attention = trainingConfig.alpha_attention
        #self.alpha_hidden_state = trainingConfig.alpha_hidden_state
        #self.alpha_soft_label = trainingConfig.alpha_soft_label
        #self.alpha_true_label = trainingConfig.alpha_true_label
        #self.alpha_cls = trainingConfig.alpha_cls
        #####
        # self.alpha_hidden_logits = trainingConfig.alpha_hidden_logits
        # self.loss_weight_schema = trainingConfig.loss_weight_schema
        
    def forward(
        self,
        t_model,
        s_model,
        inputs: torch.Tensor,
        inputs_mask: torch.Tensor,
        sentence_id: torch.Tensor,
        targets: torch.Tensor,
        # distill_table: list,
        metadistill: int
    ):
        s_output = s_model(inputs, inputs_mask, sentence_id)
        s_logits = s_output['logits']
        s_attention_map = s_output['attentions']
        s_hidden_state = s_output['hidden_states'][1:] # remove embeding output
        
        return_dict = {}
        
        if metadistill == 1:
            t_output = t_model(inputs, inputs_mask, sentence_id)
            t_logits = t_output['logits']
            t_attention_map = t_output['attentions']
            t_hidden_state = t_output['hidden_states'][1:] # remove embeding output
        else:
            with torch.no_grad():
                t_output = t_model(inputs, inputs_mask, sentence_id)
                t_logits = t_output['logits']
                t_attention_map = t_output['attentions']
                t_hidden_state = t_output['hidden_states'][1:] # remove embeding output

        # calculate attention loss
        if self.alpha_attention > 0.0:
            attention_loss = self.attention_map_loss(
                t_attention_map=t_attention_map,
                s_attention_map=s_attention_map
            )
            return_dict['attention_loss'] = self.alpha_attention * attention_loss
        
        # calculate hidden state loss
        if self.alpha_hidden_state > 0:
            hidden_state_loss = self.hidden_state_loss(
                t_hidden_state = t_hidden_state,
                s_hidden_state = s_hidden_state
            )
            return_dict['hidden_state_loss'] = self.alpha_hidden_state * hidden_state_loss

        # calculate soft loss
        if self.alpha_soft_label > 0.0:
            soft_label_loss = self.soft_label_loss(
                t_logits = t_logits,
                s_logits = s_logits,
                logits_type = 'kd'
            )
            return_dict['soft_label_loss'] = soft_label_loss * self.alpha_soft_label

        # calculate ground trueth loss
        if self.alpha_true_label > 0.0:
            true_label_loss, correct, y_hat = self.true_label_loss(
                s_logits = s_logits,
                targets = targets
            )
            return_dict['true_label_loss'] = true_label_loss * self.alpha_true_label
            return_dict['correct'] = correct
            return_dict['y_hat'] = y_hat
        
        
            
        
        return return_dict
    
    def hidden_state_loss(
        self,
        t_hidden_state,
        s_hidden_state
    ):
        average_hidden_stage = torch.zeros(t_hidden_state[0].shape).to(t_hidden_state[0].device)
        compress_rate = int( (len(t_hidden_state) + 1) / len(s_hidden_state))
        
        loss = 0.0
        count = 0
        
        for i in range(len(t_hidden_state)):
            count = count + 1
            average_hidden_stage = torch.div(average_hidden_stage, 2) # 每過一層就除2
            average_hidden_stage = torch.add(average_hidden_stage, t_hidden_state[i])
            
            if count == compress_rate:
                count = 0
                t_hidden = F.normalize(average_hidden_stage, dim=-1)
                s_hidden = F.normalize(s_hidden_state[int(i/compress_rate)], dim=-1)
                loss += F.mse_loss(t_hidden, s_hidden, reduction="mean")
                
        loss /= len(s_hidden_state)
        
        return loss
    
    def attention_map_loss(
        self,
        t_attention_map, #24
        s_attention_map  #12
    ):
        t_average_attention_map = torch.zeros(t_attention_map[0].shape).to(t_attention_map[0].device)
        s_average_attention_map = torch.zeros(s_attention_map[0].shape).to(s_attention_map[0].device)
        
        for t_att in t_attention_map:
            t_average_attention_map = torch.add(t_average_attention_map, t_att)
        t_average_attention_map = torch.div(t_average_attention_map, len(t_attention_map))

        for s_att in s_attention_map:
            s_average_attention_map = torch.add(s_average_attention_map, s_att)
        s_average_attention_map = torch.div(s_average_attention_map, len(s_attention_map))
        
        
        attention_loss = F.mse_loss(t_average_attention_map, s_average_attention_map, reduction="sum")
        
        return attention_loss
    
    def last_layer_hidden_state_loss(
        self,
        t_last_hidden_states,
        s_last_hidden_states
    ):
        hidden_state_loss = F.mse_loss(t_last_hidden_states, s_last_hidden_states, reduction="mean")
        return hidden_state_loss
    
    def soft_label_loss(
        self,
        t_logits,
        s_logits,
        logits_type:str
    ):
        if logits_type == 'mse':
            soft_loss = F.mse_loss(t_logits, s_logits)   
        elif logits_type == 'kd':
            T = 5.0

            soft_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction='batchmean'
            ) * T * T

            
 
        return soft_loss

    def true_label_loss(
        self,
        s_logits,
        targets
    ):
        y_hat = torch.argmax(s_logits, dim=-1)
        correct =  int(torch.sum(y_hat==targets))
        ce_loss = F.cross_entropy(s_logits, targets)
        return ce_loss, correct, y_hat
    
    # MetaDistill
    def s_prime_forward(
        self,
        s_model,
        inputs: torch.Tensor,
        inputs_mask: torch.Tensor,
        sentence_id: torch.Tensor,
        targets: torch.Tensor,
    ):
        s_output = s_model(inputs, inputs_mask, sentence_id)
        s_logits = s_output['logits']


        train_loss, _, _ = self.true_label_loss(s_logits, targets)
        return train_loss
    
    
    
    
    
    
    
    
    
    
    
    # for Early Exit
    def layer_logits_loss(
        self,
        s_layer_logits_output,
        targets,
        training_flag,
        loss_weight_schema:str # asc, dsc
    ):
        layer_ce_loss = 0.0
        w = 1
        total_weights = 0
        if training_flag:
            for i in range(len(s_layer_logits_output)-1): ## skip last layer
                if loss_weight_schema == 'asc':
                    w = (i + 1)
                elif loss_weight_schema == 'dsc':
                    w = (len(s_layer_logits_output) - i - 1)
                layer_ce_loss += w * self.ce_loss(s_layer_logits_output[i], targets)
                total_weights += w
            return layer_ce_loss
        
        
        
    
import time

import argparse
import torch
import torch.nn as nn
from header2.model.utils import fix_seed, load_data, EarlyStopping
from header2.model import GLUERunner
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import torch.nn.functional as F
from header2.data import GLUEDataset
import copy
from torchinfo import summary

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 2e-5)
    parser.add_argument('--device', type = str, default = 'cuda:3')
    parser.add_argument('--trainable', type = int, default = 1)
    parser.add_argument('--patience', type = int, default = 5)
    parser.add_argument('--delta', type = float, default = 1e-3)

    #---------- load weight --------
    parser.add_argument('--load', type = str, default = None)

    # model
    parser.add_argument('--label', type = int, default = 2, choices = [2,3,5])
    parser.add_argument('--mask', type = float, default = 0.0)
    parser.add_argument('--type', type = str, default = 'accuracy', choices = ['accuracy', 'loss'])
    # -------- hyperparameter BERT with convolution --------
    parser.add_argument('--seq_len', type = int, default = 128)
    
    args = parser.parse_args()

    print('=' * 70)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 70)

    return args

def main() -> None:
    args = parse()
    
    # -------- Prepare dataloader --------
    trainset = GLUEDataset(
        task = 'mrpc',
        max_length = args.seq_len,
        ckpt = "bert-base-uncased",
        flag = 'train',
        part = 1,
        rand_mask_part=0.0
    )
               
    testset = GLUEDataset(
        task = 'mrpc',
        max_length = args.seq_len,
        ckpt = "bert-base-uncased",
        flag = 'test',
        part = 1,
        rand_mask_part=0.0
    )
        
    train_loader = load_data(trainset, batch_size = args.batch_size, shuffle = True)
    test_loader = load_data(testset, batch_size = args.batch_size, shuffle = False)
    
    
    # -------- Prepare model --------
    
    teacher_config = BertConfig(
        attention_probs_dropout_prob = 0.1,
        hidden_act = "gelu",
        hidden_dropout_prob = 0.1,
        hidden_size = 768,
        initializer_range = 0.02,
        intermediate_size = 3072,
        layer_norm_eps = 1e-12,
        max_position_embeddings = 512,
        model_type = "bert",
        num_attention_heads = 12,
        num_hidden_layers = 12,
        pad_token_id = 0,
        position_embedding_type = "absolute",
  
        type_vocab_size = 2,
        vocab_size = 30522
    )
    teacher = BertForSequenceClassification(teacher_config).from_pretrained(
        "bert-base-uncased",
        num_labels = args.label,
        output_hidden_states = True,
        output_attentions = True
    )
    
    path = './ckpt/BERT/teacher_mrpc.pt'
    state_dict = torch.load(path,map_location='cpu')
    teacher.load_state_dict(state_dict)
    
    #summary(teacher, (2, 512))
    print(teacher)
    for i,param in enumerate(teacher.parameters()):
        param.requires_grad = False
    
    
    student_config = BertConfig(
        attention_probs_dropout_prob = 0.1,
        hidden_act = "gelu",
        hidden_dropout_prob = 0.1,
        hidden_size = 768,
        initializer_range = 0.02,
        intermediate_size = 3072,
        layer_norm_eps = 1e-12,
        max_position_embeddings = 512,
        model_type = "bert",
        num_attention_heads = 12,
        num_hidden_layers = 6,
        pad_token_id = 0,
        num_labels = args.label,
        position_embedding_type = "absolute",
        type_vocab_size = 2,
        vocab_size = 30522,
        output_hidden_states = True,
        output_attentions = True
    )
    student = BertForSequenceClassification(student_config)
    
    save_path = './ckpt/BERT/student_mrpc.pt'

    #####

    # copy teacher for student
    # encoder layer use "two out of one" strategy
    # copy embedding layer
    # hint do not make the teacher and student use the same memory in intermediate layer

    #####
    student.bert.embeddings = copy.deepcopy(teacher.bert.embeddings)
    
    for i in range(6):
        student.bert.encoder.layer[i] = copy.deepcopy(teacher.bert.encoder.layer[2*i + 1])
    # open grad
    for param in student.parameters():
        param.requires_grad = True 
    
    if args.load:
        student.load_state_dict(torch.load(args.load))

 

    # --------- Optimizer -----------
    optimizer = AdamW(student.parameters(), lr = args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 5, args.epochs)
    criterion = nn.CrossEntropyLoss()


    model_ckpt = [
        EarlyStopping(
            patience = args.patience,
            verbose = True,
            delta = args.delta,
            path = save_path
        )
    ]

    runner = GLUERunner(
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        criterion=criterion,
        model_ckpt=model_ckpt[0],
        device=args.device,
    )

    if args.trainable == 1:
        print(f'Start to train...\n')
        start = time.time()
        runner.train(
            args.epochs,
            train_loader = train_loader, 
            valid_loader = test_loader,
            scheduler = scheduler,
            save_type='accuracy'
        )
        # runner.val(test_loader)
        end = time.time()
        print(f'End in {end - start:.4f}s...\n')
    else:
        runner.test(test_loader)


if __name__ == '__main__':
    fix_seed(seed=87)
    
    main()
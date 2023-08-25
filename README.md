## How to use it
1. Directly training a small BERT(6-layer encoder layer)   
```python3 main_BERT.py```

2. Use the teacher model(12-layer) weight to initial the student model and training    
* One out of two   
```python3 main_BERT2.py```


3. Knowledge Distillation    
* Intermediate hidden state (ClS token)   
* Output distribution     
```python3 main_BERT3.py```

## Result
1. Directly training a small BERT(6-layer encoder layer)
<img src="https://i.imgur.com/LRPxlrq.png">
2. Use the teacher model(12-layer) weight to initial the student model and training
* One out of two
<img src=https://i.imgur.com/hwMnMuI.png>
3. Knowledge Distillation
* Intermediate hidden state (ClS token)
* Output distribution
<img src="https://i.imgur.com/khPdu0b.png">
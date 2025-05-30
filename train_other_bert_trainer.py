import torch
import torch.nn as nn
from transformers import ModernBertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error
import torch.distributed as dist
import numpy as np
import os
import random
import wandb
# 解决tokenizer并行问题，huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

label_mapping = {
    "其它": 0, "政务": 1, "娱乐": 2, "科学": 3, "教育": 4, "时政": 5, "新闻": 6, "农业": 7,
    "房地产": 8, "douban": 9, "法律1": 10, "经济": 11, "法律": 12, "学习强国": 13, "电力": 14,
    "政府工作报告": 15, "外交": 16, "企业": 17
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=8192
    )
    tokenized["label"] = examples["label"]
    tokenized["score"] = examples["score"]
    return tokenized

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = label_mapping[item['label']]
        score = item['score']
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.float)
        }



class CustomModel(nn.Module):
    def __init__(self, model_name="neavo/modern_bert_multilingual"):
        super().__init__()
        self.bert = ModernBertForSequenceClassification.from_pretrained(model_name, num_labels=18)
        self.regression_head = nn.Linear(18, 1)
        
        # 启用梯度检查点
        self.bert.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, labels=None, score=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits
        regression_output = self.regression_head(logits)
        
        loss = None
        if labels is not None and score is not None:
            cls_loss = nn.CrossEntropyLoss()(logits, labels)
            reg_loss = nn.MSELoss()(regression_output, score.view(-1, 1))
            loss = 0.5 * cls_loss + 0.5 * reg_loss
        
        # 返回元组而不是字典，使其与Trainer兼容
        return (loss, logits, regression_output) if loss is not None else (logits, regression_output)




"""
出现错误：
[rank0]:   File "/root/for_may_conference/remake/train_other_bert/train_other_bert_trainer.py", line 191, in <module>
[rank0]:     print("开始训练...")
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 2241, in train
[rank0]:     return inner_training_loop(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 2612, in _inner_training_loop
[rank0]:     self._maybe_log_save_evaluate(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 3085, in _maybe_log_save_evaluate
[rank0]:     metrics = self._evaluate(trial, ignore_keys_for_eval)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 3039, in _evaluate
[rank0]:     metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 4105, in evaluate
[rank0]:     output = eval_loop(
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/trainer.py", line 4394, in evaluation_loop
[rank0]:     metrics = self.compute_metrics(
[rank0]:   File "/root/for_may_conference/remake/train_other_bert/train_other_bert_trainer.py", line 85, in compute_metrics
[rank0]:     logits, regression_outputs = eval_pred.predictions
[rank0]: ValueError: too many values to unpack (expected 2)
"""
# class CustomModel(nn.Module):
#     def __init__(self, model_name="neavo/modern_bert_multilingual"):
#         super().__init__()
#         self.bert = ModernBertForSequenceClassification.from_pretrained(model_name, num_labels=18)
#         self.regression_head = nn.Linear(18, 1)
        
#         # 启用梯度检查点
#         self.bert.gradient_checkpointing_enable()

#     def forward(self, input_ids, attention_mask, labels=None, score=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         logits = outputs.logits
#         regression_output = self.regression_head(logits)
        
#         loss = None
#         if labels is not None and score is not None:
#             cls_loss = nn.CrossEntropyLoss()(logits, labels)
#             reg_loss = nn.MSELoss()(regression_output, score.view(-1, 1))
#             loss = 0.5 * cls_loss + 0.5 * reg_loss
            
#         return {
#             'loss': loss,
#             'logits': logits,
#             'regression_output': regression_output
#         }

# def compute_metrics(eval_pred):
#     logits, regression_outputs = eval_pred.predictions
#     labels, true_scores = eval_pred.label_ids
    
#     # 分类指标
#     predictions = np.argmax(logits, axis=1)
#     accuracy = (predictions == labels).mean()
    
#     # 回归指标
#     regression_outputs = regression_outputs.flatten()
#     mse = ((regression_outputs - true_scores) ** 2).mean()
#     mae = mean_absolute_error(true_scores, regression_outputs)
    
#     return {
#         "accuracy": accuracy,
#         "mse": mse,
#         "mae": mae
#     }

def compute_metrics(eval_pred):
    # 预测是一个元组，包含logits和regression_outputs
    predictions = eval_pred.predictions
    
    if not isinstance(predictions, tuple):
        predictions = (predictions,)
    
    if len(predictions) >= 2:
        logits = predictions[0]
        regression_outputs = predictions[1]
    else:
        logits = predictions[0]
        regression_outputs = np.zeros((len(logits), 1))
    
    labels = eval_pred.label_ids
    
    # 如果labels是元组，解包它
    if isinstance(labels, tuple) and len(labels) >= 2:
        class_labels, true_scores = labels
    else:
        class_labels = labels
        true_scores = np.zeros_like(regression_outputs)
    
    # 计算分类指标
    class_predictions = np.argmax(logits, axis=1)
    accuracy = (class_predictions == class_labels).mean()
    
    # 计算回归指标
    regression_outputs = regression_outputs.flatten()
    true_scores = true_scores.flatten() if hasattr(true_scores, 'flatten') else true_scores
    mse = ((regression_outputs - true_scores) ** 2).mean()
    mae = mean_absolute_error(true_scores, regression_outputs)
    
    return {
        "accuracy": accuracy,
        "mse": mse,
        "mae": mae
    }


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    set_seed(3407)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # 只在主进程上初始化wandb
    if local_rank == 0:
        wandb.init(project="train_other_bert_trainer_final", name="train_other_bert_trainer_final")
        report_to = "wandb"  # 主进程使用wandb
    else:
        report_to = "none"


    
    model_name = "neavo/modern_bert_multilingual"
    
    dataset = load_dataset('json', data_files='/root/for_may_conference/remake/data_for_train_bert_need_shuffle.json')
    shuffled_dataset = dataset["train"].shuffle(seed=3407)
    train_test_split = shuffled_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    train_test_split = train_dataset.train_test_split(test_size=0.05)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    if local_rank == 0:
        print('数据集拆分完成')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    train_dataset = CustomDataset(tokenized_train)
    eval_dataset = CustomDataset(tokenized_eval)
    test_dataset = CustomDataset(tokenized_test)
    
    if local_rank == 0:
        print('数据集预处理完成')
        print(f'使用设备: cuda:{local_rank}, 可用的GPU数量: {torch.cuda.device_count()}')

    model = CustomModel()
    
    # 启用梯度检查点
    model.bert.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        learning_rate=2e-5,
        per_device_train_batch_size=2, 
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        logging_steps=50,
        report_to=report_to,
        run_name="train_other_bert_trainer_final",
        # 分布式训练参数
        local_rank=local_rank,           
        fp16=True,                        
        gradient_accumulation_steps=4,    # 梯度累积，减少GPU内存使用
        dataloader_num_workers=4,         
        ddp_find_unused_parameters=False, # 提高DDP效率
    )
    class CustomDataCollator:
        def __call__(self, features):
            batch = {}
            batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
            batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
            batch["labels"] = torch.stack([f["labels"] for f in features])
            batch["score"] = torch.stack([f["score"] for f in features])
            return batch
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator(),
    )

    if local_rank == 0:
        print("开始训练...")
    trainer.train()

    if local_rank == 0:
        print("测试集评估...")
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")
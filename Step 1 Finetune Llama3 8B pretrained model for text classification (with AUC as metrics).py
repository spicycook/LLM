# -*- coding: utf-8 -*-
"""
Author: Yue Tang
Date: 2025-01-23
Time: 17:08 EST
Research Interests: Supply Chain Management, Sustainability, Healthcare Operations
PhD Candidate in Information Systems and Operations Management
Goizueta Business School
Emory University
"""


# !huggingface-cli login
from huggingface_hub import login
access_token = "xxx" # place your huggingface token here
login(token=access_token)

""" Imports"""
import os, random, functools, csv
import re
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


""" Introduction"""
'''
In this script, my data have two columns: fulltext, labels. 
You only need to replace the column names by yours.
The GPU used is Nvidia A100 80GB.
You may need to reduce batch_size, MAX_LEN as well as other parameters to cater to your config.

Note I am using a 8bit quantized model for fine tuning.
'''


""" Parameters"""
random_state = 20250123
train_size = 0.7 # 70% data used for training
val_test = 1 # val: test = 1:1
""" What does LoRA mean? See below: W^{hat} = W+ lora_alpha/r (A x B), where W is dxd, A is  dxr, and B is rxd."""
r = 64 # the dimension of the low-rank matrices/ was 16/ 32/ 64
lora_alpha = 32 # scaling factor for LoRA activations vs pre-trained weight activations; previous: 8, 16
MAX_LEN = 2048 # was 512, 1024, 2048, 1024
batch_size = 32 # was 32/ 16/ 64
num_train_epochs = 25 # was 2, 5, 10, 20, 10, 20, 30, 50


""" Load Factiva news"""
df = pd.read_excel("your_data.xlsx")

# Select only the relevant columns from both dataframes
df = df[['fulltext', 'label']]
# Preprocess
def preprocess_text(text):
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r'<.*?>', '', text)     # Remove HTML tags (if any)
    text = ''.join(c for c in text if c.isprintable()) # Remove non-printable characters
    text = re.sub(r'[!?.;]{2,}', lambda x: x.group(0)[0], text) # Remove excessive punctuation and semicolons
    text = " ".join(text.split()) # Remove multiple spaces (replace with a single space)
    return text
df['fulltext'] = df['fulltext'].apply(preprocess_text)

""" train_test_split"""
labels = df['label']
df_train, df_temp, labels_train, labels_temp = train_test_split(df, labels, train_size=train_size, stratify=labels, random_state=random_state)
df_val, df_test, labels_val, labels_test = train_test_split(df_temp, labels_temp, test_size=1/(val_test+1), stratify=labels_temp, random_state=random_state)
print(df_train.shape, df_val.shape, df_test.shape)

""" Convert from Pandas DataFrame to Hugging Face Dataset"""
# pandas df to HF dataset
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)
dataset_test = Dataset.from_pandas(df_test)
dataset_train_shuffled = dataset_train.shuffle(seed=random_state)  # Using a seed for reproducibility
dataset = DatasetDict({
    'train': dataset_train_shuffled,
    'val': dataset_val,
    'test': dataset_test
})

# Class weights to handle imbalance
df_train.label.value_counts(normalize=True)
class_weights=(1/df_train.label.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()

""" Load LLama model with 8 bit quantization"""
model_name = "meta-llama/Meta-Llama-3-8B"
# Enable dynamic GPU memory allocation
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

""" Quantization Config (for QLORA)"""
quantization_config = BitsAndBytesConfig(
    load_in_8bit = True, # enable 8-bit quantization
)

""" LoRA Config"""
lora_config = LoraConfig(
    r = r, 
    lora_alpha = lora_alpha, 
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

""" Load model"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    num_labels=2
)

""" Quantization: QLoRA = Quatization + LoRA"""
model = prepare_model_for_kbit_training(model)

"""* get_peft_model prepares a model for training with a PEFT method such as LoRA by wrapping the base model and PEFT configuration with get_peft_model"""
model = get_peft_model(model, lora_config)

""" Load the tokenizer"""
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

""" Update some model configs"""
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

sentences = df_test.fulltext.tolist()
all_outputs = []
# Process the sentences in batches
for i in range(0, len(sentences), batch_size):
    batch_sentences = sentences[i:i + batch_size]
    inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        all_outputs.append(outputs['logits'])

""" Concatenate all outputs into a single tensor"""
final_outputs = torch.cat(all_outputs, dim=0)
final_outputs.argmax(axis=1)

"""* Move to CPU so we can use numpy and set prediction colum to it"""
df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
df_test['predictions'].value_counts()

""" Analyze performance as in intro notebook"""
def get_performance_metrics(df_test):
    y_test = df_test.label
    y_pred = df_test.predictions
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_pred)
    print("AUC Score:", auc)

print("------------------Before Training------------------")
get_performance_metrics(df_test)

col_to_delete = ['fulltext']
def llama_preprocessing_function(examples):
    return tokenizer(examples['fulltext'], truncation=True, max_length=MAX_LEN)
tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
tokenized_datasets.set_format("torch")

""" Data Collator"""
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

""" Evaluation Metrics"""
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 1]  # Get the predicted probabilities for the positive class (class 1)
    auc = roc_auc_score(labels, predictions)
    return {'auc': auc}

""" Custom Trainer"""
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

""" Training Args"""
training_args = TrainingArguments(
    output_dir = 'your_output_folder',
    learning_rate = 5e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = num_train_epochs,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model = 'auc',  # Set this to use AUC for model selection
    greater_is_better = True  # AUC higher is better
)

""" Trainer"""
trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    class_weights=class_weights,
)
"""* For more info, check this website: https://huggingface.co/docs/transformers/en/training

### Traiing"""
train_result = trainer.train()

def make_predictions(model, df_test, batch_size):
    sentences = df_test.fulltext.tolist()
    all_outputs = []
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
    final_outputs = torch.cat(all_outputs, dim=0)
    df_test['predictions'] = final_outputs.argmax(axis=1).cpu().numpy()

# Call the function

print("------------------After Training------------------")
make_predictions(model, df_test, batch_size)
get_performance_metrics(df_test)

""" Saving the model trainer state and model adapters"""
metrics = train_result.metrics
max_train_samples = len(dataset_train)
metrics["train_samples"] = min(max_train_samples, len(dataset_train))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
""" Save model"""
trainer.save_model("your_saved_model")

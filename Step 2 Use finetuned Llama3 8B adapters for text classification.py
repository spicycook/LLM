"""
Author: Yue Tang
Date: 2025-01-23
Time: 17:08 EST
Research Interests: Supply Chain Management, Sustainability, Healthcare Operations
PhD Candidate in Information Systems and Operations Management
Goizueta Business School
Emory University
"""


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy
import re
from peft import PeftModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import accelerate
import bitsandbytes

# !huggingface-cli login
from huggingface_hub import login
access_token = "XXX"
login(token=access_token)

# Load new data
import pandas as pd
df = pd.read_excel("your_data.xlsx")
df = df[['fulltext','doc_id']]  # Subset to keep only "fulltext" and 'doc_id'

# Preprocess
def preprocess_text(text):
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags (if any)
    text = ''.join(c for c in text if c.isprintable())  # Remove non-printable characters
    text = re.sub(r'[!?.;]{2,}', lambda x: x.group(0)[0], text)  # Remove excessive punctuation and semicolons
    text = " ".join(text.split())  # Remove multiple spaces
    return text
df['fulltext'] = df['fulltext'].apply(preprocess_text)

# Download base model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "meta-llama/Meta-Llama-3-8B"
# finetuned model
adapter_path = r"your_saved_adapters_path"
model = AutoModelForSequenceClassification.from_pretrained(model_name , num_labels=2)
model.load_adapter(peft_model_id=adapter_path, adapter_name="my_adapter")

# setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# make inference
def make_predictions(model,df):
  sentences = df.fulltext.tolist()
  batch_size = 16  # You can adjust this based on your system's memory capacity
  all_outputs = []
  for i in range(0, len(sentences), batch_size):
      batch_sentences = sentences[i:i + batch_size]
      inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    #   inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}
      inputs = {k: v.to(device) for k, v in inputs.items()}
      with torch.no_grad():
          outputs = model(**inputs)
          all_outputs.append(outputs['logits'])
  final_outputs = torch.cat(all_outputs, dim=0)
  df['predictions']=final_outputs.argmax(axis=1).cpu().numpy()

model.to(device)  # Move model to GPU
make_predictions(model=model, df=df)
model.to("cpu")
torch.cuda.empty_cache()

df.to_excel("save_results.xlsx", index=False)  # Set index=False to avoid including the index column in the Excel file


'''
There is no performance check here, since this dataset is unlabelled.

'''

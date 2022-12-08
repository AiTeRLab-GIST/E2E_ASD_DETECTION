import torch
from transformers import TrainingArguments

train = False
evaluate = True
feat_ext = False
infer = False

inputs = 'path'
outputs = 'asd'
batch_size = 128

db_type = 'db_type'

save_path = 'save_path'
wav_path = f'{save_path}/{db_type}'
df_names = ['data/train.csv', 'data/valid.csv', 'data/eval.csv']

model_name = 'facebook/wav2vec2-base-960h'
fix_feat_ext = False 
condition = 'fix-fe' if fix_feat_ext else 'tune-fe'
exp_dir = './exp/'

pooling_mode = 'mean'
is_regression = False
sampling_rate = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

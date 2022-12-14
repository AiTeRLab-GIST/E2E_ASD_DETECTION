import argparse
import pdb
import numpy as np
import os
import sys
#os.environ["NCCL_DEBUG"] = "INFO"
from transformers import AutoConfig, Wav2Vec2Processor, TrainingArguments
from functools import partial

import torch

import conf
from trainer import CTCTrainer as Trainer
from model import Wav2Vec2ForSpeechClassification as Model
from dataprocessor import DataProcessor
from utils import preprocess_function, compute_metrics, speech_batch_to_array_fn, predict, feat_ext
from dataloader import DataCollatorCTCWithPadding as Collator
from datasets import load_dataset, load_metric

torch.set_num_threads(16)

def process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--infer', action='store_true', default=False)
    parser.add_argument('--exp', type=str, default=None)

    args = parser.parse_args()

    if args.target_model == 'ft':
        model.freeze_feature_extractor()

    exp_base = f'libri_{args.exp}_ft'

    exp_name = os.path.join(conf.exp_dir,exp_base)

    training_args = TrainingArguments(
                    output_dir=exp_name,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    gradient_accumulation_steps=16,
                    evaluation_strategy="steps",
                    load_best_model_at_end=True,
                    num_train_epochs=10.0,
                    fp16=True,
                    save_steps=10,
                    eval_steps=10,
                    logging_dir='./logs',
                    logging_steps=10,
                    learning_rate=1e-4,
                    save_total_limit=10)
    
    
    dbprocer = DataProcessor()
    train_dataset, eval_dataset, config, label_list = dbprocer.load_dataset_and_config()
    processor = Wav2Vec2Processor.from_pretrained(conf.model_name)

    data_collator = Collator(processor=processor, padding=True)

    model = Model.from_pretrained(conf.model_name, config=config)

    if args.train:
        train_dataset = train_dataset.map(partial(partial(preprocess_function, label_list = label_list),processor = processor), batch_size = conf.batch_size, batched = True, num_proc = 1)
        eval_dataset = eval_dataset.map(partial(partial(preprocess_function, label_list = label_list),processor = processor), batch_size = conf.batch_size, batched = True, num_proc = 1)
        trainer = Trainer(model = model, data_collator = data_collator, args = training_args,
                          compute_metrics = compute_metrics, train_dataset = train_dataset,
                          eval_dataset = eval_dataset, tokenizer = processor.feature_extractor)
        trainer.train()
        processor.save_pretrained('test')

    if args.eval: 
        exps = os.listdir(exp_name)
        exp_idx = np.argmax([int(exp.split('-')[1]) for exp in exps])
        exp_name = os.path.join(exp_name, exps[exp_idx])

        config = AutoConfig.from_pretrained(exp_name)
        processor = Wav2Vec2Processor.from_pretrained('test')
        model = Model.from_pretrained(exp_name).to(conf.device)
        import librosa
        from sklearn.metrics import classification_report
        test_dataset = load_dataset("csv", data_files={"test": f"{conf.save_path}/{conf.df_names[2]}"}, delimiter="\t")["test"]
        test_dataset = test_dataset.map(speech_batch_to_array_fn)
        
        result = test_dataset.map(partial(partial(predict, processor = processor),model = model), batched = True, batch_size = 8)
        
        label_names = [config.id2label[i] for i in range(config.num_labels)]
        y_true = [{1:0, 2:1}[name] for name in result[conf.outputs]]
        y_pred = result["predicted"]
        
        report = classification_report(y_true, y_pred, digits=4)
        print(report)

if __name__ == '__main__':
    process()

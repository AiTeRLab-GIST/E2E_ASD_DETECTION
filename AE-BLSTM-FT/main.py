import os
import numpy as np
import torch
from tqdm import tqdm
import conf
import utils 

from trainer import Trainer
from dataset import egemaps_dataset as Dataset
from datasets import load_dataset, load_metric

from load_datasets import load_datasets

from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--target_model', type=str, default = 'rgrs')
parser.add_argument('--exp', type=str)

args = parser.parse_args()

device = torch.device('cuda')

def process():
    train_df, valid_df, test_df, label_list = load_datasets()

    exp_name = args.exp
    
    if args.target_model == 'rgrs':
        from model import MultiTaskAutoEncoder as Model
        model = Model().cuda()
    if args.target_model == 'clsf':
        from model import AEBLSTMFT as Model
        model = Model().cuda()
        if args.train or args.featext:
            ae_exp = utils.get_part_model(exp_name = exp_name, part = 'rgrs')
            aedata = torch.load(ae_exp)
            model.AEPart.load_state_dict(aedata['model_state_dict'])
        elif args.eval:
            exp = utils.get_part_model(exp_name = exp_name, part = 'clsf')
            model_data = torch.load(exp)
            model.load_state_dict(model_data['model_state_dict'])
            print(f'Start evaluation on experiment:{exp}')

    train_dataset = Dataset(train_df)
    valid_dataset = Dataset(valid_df)
    test_dataset = Dataset(test_df)

    if args.train:
        if not os.path.isdir(os.path.join(conf.exp_dir, exp_name)):
            os.makedirs(os.path.join(conf.exp_dir, exp_name), exist_ok=True)


        train_dataloader = DataLoader(train_dataset, batch_size = conf.batch_size, shuffle=1)
        valid_dataloader = DataLoader(valid_dataset, batch_size = conf.batch_size, shuffle=1)

        trainer = Trainer(model = model,
                          train_dataloader = train_dataloader,
                          valid_dataloader = valid_dataloader,
                          target_model = args.target_model,
                          exp_name = exp_name
                          )
        trainer.train()

    if args.eval:
        model.eval()
        test_dataloader = DataLoader(test_dataset, batch_size = conf.batch_size)
        import librosa
        from sklearn.metrics import classification_report
        test_dataset = load_dataset("csv", data_files={"test": f"{conf.save_path}/{conf.df_names[2]}"}, delimiter="\t")["test"]
        test_result = []
        true_labels = []
        results = []
        for test_batch in tqdm(test_dataloader, desc='evaluation steps'):
            inputs, labels = test_batch
            outputs = model(inputs.type(torch.float).to(conf.device))
            for idx, output in enumerate(outputs[2]):
                if output[0] > output[1]:
                    results.append(0)
                else:
                    results.append(1)
            true_labels.append(labels)
        true_labels = np.concatenate(true_labels) - 1
        
        report = classification_report(true_labels, results, digits=4)
        print(report)

if __name__ == '__main__':
    process()
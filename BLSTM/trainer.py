from datetime import datetime as dt
from tqdm import tqdm
import conf
import pdb
import torch
from torch import nn
import traceback
import numpy as np
from typing import Any, Dict, Union
from packaging import version

now = dt.now()

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, target_model, exp_name):
        self.model = model
        self.n_epoches = 1000
        self.rgrs_criterion = nn.MSELoss()
        self.clsf_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.9, patience = 20)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.target_model = target_model
        self.exp_name = exp_name

    def compute_loss(self, inputs, outputs, labels, w_rgr, w_clsf):
        rgrs, z, logits = outputs
        labels = labels-1
        if self.target_model == 'rgrs':
            try:
                loss = 0.9*self.rgrs_criterion(inputs.reshape(-1,88), rgrs) + 0.1*self.clsf_criterion(logits, labels.type(torch.long).to(conf.device))
            except:
                pdb.set_trace()
        elif self.target_model == 'clsf':
            #pdb.set_trace()
            loss = self.clsf_criterion(logits, labels.type(torch.long).to(conf.device))
        elif self.target_model == 'joint':
            loss = w_rgr*self.rgrs_criterion(inputs.reshape(-1,88), rgrs) + w_clsf*self.clsf_criterion(logits, labels.type(torch.long).to(conf.device))
        return loss

    def save_model(self, epoch, model, optim):
        exp_base = f'ae-blstm-{self.target_model}'
        try:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        }, f'{conf.exp_dir}/{self.exp_name}/{exp_base}_ep{epoch:04d}.pt')
            print(f'model saved on {conf.exp_dir}/{exp_base}_ep{epoch:04d}.pt')
        except:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optim.state_dict(),
                        }, f'{conf.exp_dir}/{self.exp_name}/{exp_base}_ep{epoch:04d}.pt')
            print(f'model saved on {conf.exp_dir}/{self.exp_name}/{exp_base}_ep{epoch:04d}.pt')

    def train(self):
        train_dataloader = self.train_dataloader
        valid_dataloader = self.valid_dataloader
        model = self.model
        patience = 0
        w_rgr, w_clsf = 1, 0
        for epoch in range(self.n_epoches):
            w_rgr = 1 * np.exp(-0.05 * epoch)
            w_clsf = 1 - w_rgr
            for train_batch in tqdm(train_dataloader, desc='training steps'):
                model.train()
                
                inputs, labels = train_batch

                inputs = inputs.type(torch.float).to(conf.device)
                outputs = model(inputs)
                
                self.optimizer.zero_grad()
                try:
                    loss = self.compute_loss(inputs, outputs, labels, w_rgr, w_clsf)
                except Exception as E:
                    print(E)
                    pdb.set_trace()
                    traceback.print_exc()
                loss.backward()
                self.optimizer.step()
            val_loss = self.validate(model, valid_dataloader, w_rgr, w_clsf)
            self.lr_scheduler.step(val_loss)
            if epoch == 0:
                min_val_loss = val_loss
            elif val_loss < min_val_loss:
                print(f'validation loss on epoch {epoch+1}, val_loss improved from {min_val_loss} to {val_loss}')
                min_val_loss = val_loss
                patience = 0
                self.save_model(epoch, model, self.optimizer)
            else:
                patience += 1
                if patience == 100:
                    break
                elif epoch == self.n_epoches - 1:
                    self.save_model(epoch, model, self.optimizer)
                    break

    def validate(self, model, valid_loader, w_rgr, w_clsf):
        model.eval()
        val_loss = []
        for val_batch in valid_loader:
            inputs, labels = val_batch
            inputs = inputs.type(torch.float).to(conf.device)
            outputs = model(inputs)

            val_loss.append(self.compute_loss(inputs, outputs, labels, w_rgr, w_clsf))

        return torch.mean(torch.tensor(val_loss))

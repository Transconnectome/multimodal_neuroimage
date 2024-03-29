from torch.nn import MSELoss,L1Loss,BCELoss, BCEWithLogitsLoss, Sigmoid
from losses import Percept_Loss, Cont_Loss, Mask_Loss, Merge_Loss, UNet_Loss
import csv
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from itertools import zip_longest
from metrics import Metrics
import torch

import wandb
import time

class Writer():
    """
    main class to handle logging the results, both to tensorboard and to a local csv file # so where is csv file..^^?
    """
    def __init__(self,sets,val_threshold,**kwargs):
        self.register_args(**kwargs)
        self.register_losses(**kwargs)
        self.create_score_folders()
        self.metrics = Metrics()
        self.current_metrics = {}
        self.sets = sets
        self.val_threshold = val_threshold
        self.total_train_steps = 0
        self.eval_iter = 0
        self.subject_accuracy = {}
        self.tensorboard = SummaryWriter(log_dir=self.tensorboard_dir, comment=self.experiment_title)
        for set in sets:
            setattr(self,'total_{}_loss_values'.format(set),[])
            setattr(self,'total_{}_loss_history'.format(set),[])
        for name, loss_dict in self.losses.items():
            if loss_dict['is_active']:
                for set in sets:
                    setattr(self, '{}_{}_loss_values'.format(name,set),[])
                    setattr(self, '{}_{}_loss_history'.format(name,set),[])

    def create_score_folders(self):
        self.tensorboard_dir = Path(os.path.join(self.log_dir, self.experiment_title))
        self.csv_path = os.path.join(self.experiment_folder, 'history')
        os.makedirs(self.csv_path, exist_ok=True)
        if self.task == 'fine_tune' or 'bert' or 'test':
            self.per_subject_predictions = os.path.join(self.experiment_folder, 'per_subject_predictions')
            os.makedirs(self.per_subject_predictions, exist_ok=True)

    def save_history_to_csv(self):
        rows = [getattr(self, x) for x in dir(self) if 'history' in x and isinstance(getattr(self, x), list)]
        column_names = tuple([x for x in dir(self) if 'history' in x and isinstance(getattr(self, x), list)])
        export_data = zip_longest(*rows, fillvalue='')
        with open(os.path.join(self.csv_path, 'full_scores.csv'), 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(column_names)
            wr.writerows(export_data)


    def loss_summary(self,lr):
        self.scalar_to_tensorboard('learning_rate',lr,self.total_train_steps)
        loss_d = self.append_total_to_losses()
        for name, loss_dict in loss_d.items():
            if loss_dict['is_active']:
                for set in self.sets:
                    title = name + '_' + set # name : binary classification & total
                    values = getattr(self,title + '_loss_values')
                    if len(values) == 0:
                        continue
                    score = np.mean(values)
                    history = getattr(self,title + '_loss_history')
                    history.append(score)
                    print('{}: {}'.format(title,score))
                    setattr(self,title + '_loss_history',history)
                    self.scalar_to_tensorboard(title,score)

    def accuracy_summary(self,mid_epoch):
        pred_all_sets = {x:[] for x in self.sets}
        truth_all_sets = {x:[] for x in self.sets}
        metrics = {}
        for subj_name,subj_dict in self.subject_accuracy.items():
            
            if self.fine_tune_task == 'binary_classification':
                subj_dict['score'] = torch.sigmoid(subj_dict['score'].float())

            # subj_dict['score'] denotes the logits for sequences for a subject
            subj_pred = subj_dict['score'].mean().item() 
            subj_error = subj_dict['score'].std().item()

            subj_truth = subj_dict['truth'].item()
            subj_mode = subj_dict['mode'] # train, val, test

            with open(os.path.join(self.per_subject_predictions,'iter_{}.txt'.format(self.eval_iter)),'a+') as f:
                f.write('subject:{} ({})\noutputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_pred,subj_error,subj_truth))
            pred_all_sets[subj_mode].append(subj_pred) # don't use std in computing AUROC, ACC
            truth_all_sets[subj_mode].append(subj_truth)
            

        for (name,pred),(_,truth) in zip(pred_all_sets.items(),truth_all_sets.items()):
            sigmoid = Sigmoid()
            if len(pred) == 0:
                continue
            if self.fine_tune_task == 'regression':
                metrics[name + '_MAE'] = self.metrics.MAE(truth,pred)
                metrics[name + '_MSE'] = self.metrics.MSE(truth,pred)
                metrics[name +'_NMSE'] = self.metrics.NMSE(truth,pred)
                metrics[name + '_R2_score'] = self.metrics.R2_score(truth,pred)
                
            else:
                metrics[name + '_Balanced_Accuracy'] = self.metrics.BAC(truth,[x>0.5 for x in torch.Tensor(pred)])
                metrics[name + '_Regular_Accuracy'] = self.metrics.RAC(truth,[x>0.5 for x in torch.Tensor(pred)]) # Stella modified it
                metrics[name + '_AUROC'] = self.metrics.AUROC(truth,pred)
                metrics[name+'_best_bal_acc'], metrics[name + '_best_threshold'],metrics[name + '_gmean'],metrics[name + '_specificity'],metrics[name + '_sensitivity'],metrics[name + '_f1_score'] = self.metrics.ROC_CURVE(truth,pred,name,self.val_threshold)
            self.current_metrics = metrics
            
            
        for name,value in metrics.items():
            self.scalar_to_tensorboard(name,value)
            if hasattr(self,name):
                l = getattr(self,name)
                l.append(value)
                setattr(self,name,l)
            else:
                setattr(self, name, [value])
            print('{}: {}'.format(name,value))
        self.eval_iter += 1
        if mid_epoch and len(self.subject_accuracy) > 0:
            self.subject_accuracy = {k: v for k, v in self.subject_accuracy.items() if v['mode'] == 'train'}
        else:
            self.subject_accuracy = {}

    def register_wandb(self,epoch, lr):
        wandb_result = {}
        wandb_result['epoch'] = epoch
        wandb_result['learning_rate'] = lr

        #losses 
        loss_d = self.append_total_to_losses() 
        for name, loss_dict in loss_d.items():
            # name : perceptual, reconstruction, ...
            if loss_dict['is_active']:
                for set in self.sets:
                    title = name + '_' + set
                    wandb_result[f'{title}_loss_history'] = getattr(self,title + '_loss_history')[-1]
        #accuracy
        wandb_result.update(self.current_metrics)
        wandb.log(wandb_result)

    def write_losses(self,final_loss_dict,set):
        for loss_name,loss_value in final_loss_dict.items():
            title = loss_name + '_' + set
            #print('title of write_losses is:', title) #이게 validation 48118개에서 하나씩 계속 계산됨.
            loss_values_list = getattr(self,title + '_loss_values')
            loss_values_list.append(loss_value)
            if set == 'train':
                loss_values_list = loss_values_list[-self.running_mean_size:]
            setattr(self,title + '_loss_values',loss_values_list)

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs

    def register_losses(self,**kwargs):
        self.losses = {'intensity':
                           {'is_active':False,'criterion':L1Loss(),'thresholds':[0.9, 0.99],'factor':kwargs.get('intensity_factor')},
                       'unet':
                           {'is_active':False, 'criterion':UNet_Loss(**kwargs), 'factor':1},
                       'perceptual':
                           {'is_active':False,'criterion': Percept_Loss(**kwargs),'factor':kwargs.get('perceptual_factor')},
                       'reconstruction':
                           {'is_active':False,'criterion':L1Loss(),'factor':kwargs.get('reconstruction_factor')},
                       'contrastive':
                           {'is_active':False,'criterion': Cont_Loss(**kwargs),'factor':1}, #Stella added this
                       'merge':
                           {'is_active':False,'criterion': Merge_Loss(**kwargs),'factor':1}, #Stella added this
                       'mask':
                           {'is_active':False,'criterion': Mask_Loss(**kwargs),'factor':1}, #Stella added this
                       'binary_classification':
                           {'is_active':False,'criterion': BCEWithLogitsLoss(),'factor':1}, #originally BCELoss(). Stella changed it
                       'regression':
                           {'is_active':False,'criterion':L1Loss(),'factor':1}}  #changed from L1Loss to MSELoss and changed to L1loss again
        if 'reconstruction' in kwargs.get('task').lower():
            self.losses['perceptual']['is_active'] = True
            self.losses['reconstruction']['is_active'] = True
            if 'tran' in kwargs.get('task').lower() and kwargs.get('use_cont_loss'):
                self.losses['contrastive']['is_active'] = True
            if 'tran' in kwargs.get('task').lower() and kwargs.get('use_mask_loss'):
                self.losses['mask']['is_active'] = True
        elif kwargs.get('task').lower() in ['lowfreqbert', '2dbert', 'funcstruct']:
            if kwargs.get('use_merge_loss'):
                self.losses['merge']['is_active'] = True
            if kwargs.get('use_unet_loss'):
                self.losses['unet']['is_active'] = True
            if kwargs.get('fine_tune_task').lower() == 'regression':
                self.losses['regression']['is_active'] = True
            else:
                self.losses['binary_classification']['is_active'] = True 
        elif kwargs.get('task').lower() in ['test', 'vit', 'swinfusion']: 
            if kwargs.get('fine_tune_task').lower() == 'regression':
                self.losses['regression']['is_active'] = True
            else:
                self.losses['binary_classification']['is_active'] = True

    def append_total_to_losses(self):
        loss_d = self.losses.copy()
        loss_d.update({'total': {'is_active': True}})
        return loss_d

    def scalar_to_tensorboard(self,tag,scalar,iter=None):
        if iter is None:
            iter = self.total_train_steps if 'train' in tag else self.eval_iter
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag,scalar,iter)



from loss_writer import Writer
from learning_rate import LrHandler
from data_preprocess_and_load.dataloaders import DataHandler
import torch
import warnings
import numpy as np
from tqdm import tqdm
from model import Transformer_Net, MULTModel, SwinTransformerV2, Func_Struct_Add, Func_Struct_Transfer, Func_Struct_Cross, Transformer_Net_Two_Channels, SwinFusion
from losses import get_intense_voxels
import time
import pathlib
import os

#from rfMRI_preprocessing.data_module2 import fMRIDataModule2

#DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
import builtins

#torch AMP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# ASP
#from apex.contrib.sparsity import ASP

#from apex.optimizers import FusedAdam


class Trainer():
    """
    main class to handle training, validation and testing.
    note: the order of commands in the constructor is necessary
    """
    def __init__(self,sets,**kwargs):
        
        self.register_args(**kwargs)
        
        self.fmri_type = kwargs.get('fmri_type')
        self.fmri_multimodality_type  = kwargs.get('fmri_multimodality_type')
        self.dataset_name = kwargs.get('dataset_name')
        self.eval_iter = 0
        self.batch_index = None
        self.best_loss = 100000
        self.best_AUROC = 0
        self.st_epoch = 1
        self.model_weights_path = kwargs.get('model_weights_path')
        
        self.lr_handler = LrHandler(**kwargs)
        self.train_loader, self.val_loader, self.test_loader = DataHandler(**kwargs).create_dataloaders()
        # dm = fMRIDataModule2(
        #     data_seed=1234,
        #     dataset_name='S1200',
        #     image_path='/mnt/ssd/processed/S1200/',
        #     batch_size=4,
        #     sequence_length=20,
        #     num_workers=4,
        #     to_float=True,
        #     with_voxel_norm=True,
        #     strategy=None
        # )
        # dm.setup()
        # dm.prepare_data()
        # self.train_loader = dm.train_dataloader()
        # self.val_loader = dm.val_dataloader()
        # self.test_loader = dm.test_dataloader()
        
        
        self.create_model() # model on cpu
        self.load_model_checkpoint()
        self.set_model_device() # set DDP or DP after loading checkpoint at CPUs
        
        self.create_optimizer()
        self.lr_handler.set_schedule(self.optimizer)
        self.scaler = GradScaler() 
        
        
        self.load_optim_checkpoint()

        self.writer = Writer(sets,**kwargs) #????????? ?????? writer class??? ?????????.
        self.sets = sets
        
        self.nan_list = []

        for name, loss_dict in self.writer.losses.items():
            if loss_dict['is_active']:
                print('using {} loss'.format(name))
                setattr(self, name + '_loss_func', loss_dict['criterion'])
    
    def find_pth(self, files_Path):
        file_name_and_time_lst = []
        for f_name in os.listdir(files_Path):
            if f_name.endswith('.pth'):
                written_time = os.path.getctime(os.path.join(files_Path,f_name))
                file_name_and_time_lst.append((f_name, written_time))
        # ???????????? ???????????? ?????? 
        sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

        return sorted_file_lst
    
    def load_model_checkpoint(self):
        pths = self.find_pth(self.experiment_folder)
        if len(pths) > 0 : # if there are any checkpoints
            self.recent_pth = pths[0][0] # the most recent checkpoints
            print(f'loading checkpoint from {os.path.join(self.experiment_folder,self.recent_pth)}')
            self.state_dict = torch.load(os.path.join(self.experiment_folder,self.recent_pth),map_location='cpu') #, map_location=self.device
            self.model.load_partial_state_dict(self.state_dict['model_state_dict'],load_cls_embedding=False)
            self.model.loaded_model_weights_path = os.path.join(self.experiment_folder,self.recent_pth)

        elif self.loaded_model_weights_path: # if there are weights from previous phase
            self.recent_pth = None
            self.state_dict = torch.load(self.loaded_model_weights_path,map_location='cpu') #, map_location=self.device
            self.model.load_partial_state_dict(self.state_dict['model_state_dict'],load_cls_embedding=False)
            self.model.loaded_model_weights_path = self.loaded_model_weights_path
            
        else:
            self.recent_pth = None
            self.state_dict = None
            print('There are no checkpoints or weights from previous steps')
            
    def load_optim_checkpoint(self):
        if self.recent_pth and self.state_dict: # if there are any checkpoints
            self.optimizer.load_state_dict(self.state_dict['optimizer_state_dict'])
            self.lr_handler.schedule.load_state_dict(self.state_dict['schedule_state_dict'])
            self.optimizer.param_groups[0]['lr'] = self.state_dict['lr']
            self.scaler.load_state_dict(self.state_dict['amp_state'])
            self.st_epoch = int(self.state_dict['epoch']) + 1
            self.best_loss = self.state_dict['loss_value']
            text = 'Training start from epoch {} and learning rate {}.'.format(self.st_epoch, self.optimizer.param_groups[0]['lr'])
            if 'AUROC' in self.state_dict:
                text += 'validation AUROC - {}'.format(self.state_dict['AUROC'])
            print('Training start from epoch {} and learning rate {}.'.format(self.st_epoch, self.optimizer.param_groups[0]['lr']))
            
        elif self.state_dict:  # if there are weights from previous phase
            text = 'loaded model weights:\nmodel location - {}\nlast learning rate - {}\nvalidation loss - {}\n'.format(
                self.loaded_model_weights_path, self.state_dict['lr'],self.state_dict['loss_value'])
            if 'AUROC' in self.state_dict:
                text += 'validation AUROC - {}'.format(self.state_dict['AUROC'])
            print(text)
        else:
            pass
            
    
            
    def create_optimizer(self):
        lr = self.lr_handler.base_lr
        

        params = self.model.parameters()
        print(params)
        weight_decay = self.kwargs.get('weight_decay')
        #self.optimizer = FusedAdam(params, lr=lr, weight_decay=weight_decay)
        optim = self.kwargs.get('optim')

        self.optimizer = getattr(torch.optim,optim)(params, lr=lr, weight_decay=weight_decay)  #torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        
        # attach optimizer to cuda device.
        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda(self.gpu)

    def create_model(self):
        dim = self.train_loader.dataset.dataset.get_input_shape()
        print('self.task:', self.task) # lowfreqBERT?????? ????????????.
        if self.task.lower() == 'test':
            print('test!!!')
            if self.dataset_name in ['fMRI_timeseries', 'hcp']:
                if self.fmri_type == 'timeseries':
                    self.model = Transformer_Net(**self.kwargs)
                elif self.fmri_type in ['frequency', 'time_domain_low', 'time_domain_ultralow', 'frequency_domain_low', 'frequency_domain_ultralow']:
                    self.model = Transformer_Net(**self.kwargs)
                elif self.fmri_type == 'divided_frequency':
                    if self.model_weights_path is not None and 'DTI+sMRI' in self.model_weights_path:
                        print('test for multimodal model')
                        self.model = Func_Struct_Transfer(**self.kwargs)
                    elif self.fmri_multimodality_type == 'cross_attention':
                        self.model = MULTModel(**self.kwargs)
                    elif self.fmri_multimodality_type == 'two_channels':
                        print('two channels!')
                        self.model = Transformer_Net_Two_Channels(**self.kwargs)
            elif self.dataset_name in ['DTI', 'sMRI', 'DTI+sMRI']:
                if self.VIT_name == 'vit':
                    print('single modal VIT !!')
                    self.model = VIT(**self.kwargs)
                elif self.VIT_name == 'swinv2':
                    print('Swin transformer v2 !!')
                    self.model = SwinTransformerV2(**self.kwargs)
            elif self.dataset_name in ['struct']:
                self.model = SwinFusion(**self.kwargs)
            elif self.dataset_name in ['multimodal']:
                if self.multimodality_type == 'add':
                    self.model = Func_Struct_Add(**self.kwargs)
                elif self.multimodality_type == 'cross_attention':
                    self.model = Func_Struct_Cross(**self.kwargs)
        elif self.task.lower() == '2dbert':
            print('2DBERT!!!')
            self.model = Transformer_Net(**self.kwargs)
        elif self.task.lower() == 'vit':
            if self.VIT_name == 'vit':
                print('single modal VIT !!')
                self.model = VIT(**self.kwargs)
            elif self.VIT_name == 'swinv2':
                print('Swin transformer v2 !!')
                self.model = SwinTransformerV2(**self.kwargs)
        elif self.task.lower() == 'lowfreqbert':
            if self.fmri_multimodality_type == 'cross_attention':
                print('lowfreqBERT - Cross Attention!!!')
                self.model = MULTModel(**self.kwargs)
            elif self.fmri_multimodality_type == 'two_channels':
                print('lowfreqBERT - Two Channels!!!')
                self.model = Transformer_Net_Two_Channels(**self.kwargs)
        elif self.task.lower() == 'funcstruct':
            print('Func Struct model is running!')
            if self.multimodality_type == 'add':
                self.model = Func_Struct_Add(**self.kwargs)
            elif self.multimodality_type == 'transfer':
                self.model = Func_Struct_Transfer(**self.kwargs)
            elif self.multimodality_type == 'cross_attention':
                self.model = Func_Struct_Cross(**self.kwargs)
        elif self.task.lower() == 'swinfusion':
            print('SwinFusion for DTI and sMRI!!')
            self.model = SwinFusion(**self.kwargs)

        
        
    def set_model_device(self):
        if self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                #print('id of gpu is:', self.gpu)
                self.device = torch.device('cuda:{}'.format(self.gpu))
                torch.cuda.set_device(self.gpu)
                self.model.cuda(self.gpu)
                if self.task.lower() == '2dbert': # having unused parameter for 
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], broadcast_buffers=False) 
                else: # having unused parameter (classifier token)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu], broadcast_buffers=False, find_unused_parameters=True) 
                net_without_ddp = self.model.module
            else:
                self.device = torch.device("cuda" if self.cuda else "cpu")
                self.model.cuda()
                if self.task.lower() == '2dbert':
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model,find_unused_parameters=True) 
                else: # having unused parameter (classifier token)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model,find_unused_parameters=True) 
                model_without_ddp = self.model.module
        else:
            self.device = torch.device("cuda" if self.cuda else "cpu")
            self.model = DataParallel(self.model).to(self.device)


    def training(self):
        if self.profiling == True:
            self.nEpochs = 1
        for epoch in range(self.st_epoch,self.nEpochs + 1): 
            start = time.time()
            self.train_epoch(epoch)
            # if (not self.distributed) or self.rank == 0 :
            self.eval_epoch('val')
            
            print('______epoch summary {}/{}_____\n'.format(epoch,self.nEpochs))
            # print losses
            self.writer.loss_summary(lr=self.optimizer.param_groups[0]['lr'])
            self.writer.accuracy_summary(mid_epoch=True)
            self.writer.save_history_to_csv()
            if self.use_optuna == False:
                self.save_checkpoint_(epoch, len(self.train_loader), self.scaler) 
            if self.use_optuna:
                val_AUROC = self.get_last_AUROC()
                if val_AUROC > self.best_AUROC:
                    self.best_AUROC = val_AUROC
                self.trial.report(val_AUROC, step=epoch-1)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                # else:
                #     self.save_checkpoint_(epoch, len(self.train_loader), self.scaler) 
                    
            # else:
            #     dist.barrier()
            end = time.time()
            
            print(f'time taken to perform {epoch}: {end-start:.2f}')
        
        return self.best_AUROC, self.best_loss #validation AUROC        
    '''    
    def training(self):
        if self.profiling == True:
            self.nEpochs = 1
        for epoch in range(self.st_epoch,self.nEpochs + 1): 
            start = time.time()
            self.train_epoch(epoch)
            if (not self.distributed) or self.rank == 0 :
                self.eval_epoch('val')
                print('______epoch summary {}/{}_____\n'.format(epoch,self.nEpochs))
                self.writer.loss_summary(lr=self.optimizer.param_groups[0]['lr'])
                self.writer.accuracy_summary(mid_epoch=False)
                self.writer.save_history_to_csv()
                self.save_checkpoint_(epoch, len(self.train_loader), self.scaler) 
            # else:
            #     dist.barrier()
            end = time.time()
            
            print(f'time taken to perform {epoch}: {end-start:.2f}')
    '''            
 
    def train_epoch(self,epoch):       
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)
        self.train()

        times = []
        for batch_idx, input_dict in enumerate(tqdm(self.train_loader,position=0,leave=True)): 
            ### training ###
            #start_time = time.time()
            torch.cuda.nvtx.range_push("training steps")
            self.writer.total_train_steps += 1
            self.optimizer.zero_grad()
            if self.amp:
                torch.cuda.nvtx.range_push("forward pass")
                with autocast():
                    loss_dict, loss = self.forward_pass(input_dict)
                torch.cuda.nvtx.range_pop()
                loss = loss / self.accumulation_steps # gradient accumulation - loss??? ??? ?????? ??? ??????.
                
                torch.cuda.nvtx.range_push("backward pass")
                self.scaler.scale(loss).backward()
                
                torch.cuda.nvtx.range_pop()
                
                if  (batch_idx + 1) % self.accumulation_steps == 0: # gradient accumulation
                    # gradient clipping 
                    if self.gradient_clipping == True:
                        torch.cuda.nvtx.range_push("unscale")
                        self.scaler.unscale_(self.optimizer)
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.nvtx.range_push("gradient_clipping")
                        print('executing gradient clipping')
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)
                        torch.cuda.nvtx.range_pop()
                    
                    torch.cuda.nvtx.range_push("optimize")
                    self.scaler.step(self.optimizer)
                    #self.scaler.step(self.scheduler)
                    #self.scheduler.step() # Stella added this
                    torch.cuda.nvtx.range_pop()
                    scale = self.scaler.get_scale()
                    self.scaler.update()
                    
                    skip_lr_sched = (scale > self.scaler.get_scale())
                if not skip_lr_sched:
                    self.lr_handler.schedule_check_and_update(self.optimizer) 
            else:
                torch.cuda.nvtx.range_push("forward pass")
                loss_dict, loss = self.forward_pass(input_dict)
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("backward pass")
                loss.backward()
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("optimize")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.lr_handler.schedule_check_and_update(self.optimizer)
                #self.scheduler.step() # Stella added this
            self.writer.write_losses(loss_dict, set='train')
            
            
            #end_time = time.time()
            #print(f'times taken to execute step {batch_idx}: {end_time-start_time}')
            #times.append(end_time - start_time)
            torch.cuda.nvtx.range_pop()
            
            
            
            #for profiling, early stopping
            if self.profiling == True:
                if batch_idx == 10 : 
                    break

            if (batch_idx + 1) % self.validation_frequency == 0:
                print(f'evaluating and saving checkpoint at epoch {epoch} batch {batch_idx}')
                if (not self.distributed) or self.rank == 0 :
                    ### validation ##
                    print('validation')
                    self.eval_epoch('val')
                    self.writer.loss_summary(lr=self.optimizer.param_groups[0]['lr'])
                    self.writer.accuracy_summary(mid_epoch=True)
                    self.writer.experiment_title = self.writer.experiment_title
                    self.writer.save_history_to_csv()
                
                    self.save_checkpoint_(epoch, batch_idx, self.scaler) # validation?????? checkpoint ??????               
                    self.train()
                # else:
                #     dist.barrier()
                
    def eval_epoch(self,set):
        loader = self.val_loader if set == 'val' else self.test_loader
        #print('set is:', set)# - test??? ??? ??????
        #print('test loader is:', loader) #- ??? None??
        self.eval(set)
        with torch.no_grad():
            #times = [] 
            for batch_idx, input_dict in enumerate(tqdm(loader, position=0, leave=True)):
                # start_time = time.time()
                with autocast():
                    loss_dict, _ = self.forward_pass(input_dict)
                # end_time = time.time()
                # print('times taken to execute step {0}: {1}'.format(batch_idx,end_time-start_time))
                #times.append(end_time-start_time)
                self.writer.write_losses(loss_dict, set=set)
                if self.profiling == True:
                    if batch_idx == 10 : 
                        break
        #print('time spent for validation:',np.mean(times)) 
        
    def forward_pass(self,input_dict):
        '''
        shape of input dict is : torch.Size([4, 2, 75, 93, 81, 20])
        '''
        input_dict = {k:(v.to(self.gpu) if (self.cuda and torch.is_tensor(v)) else v) for k,v in input_dict.items()}
        if self.task.lower() in ['funcstruct', 'test']:
            if self.multimodality_type in ['cross_attention']:
                output_dict = self.model(input_dict['fmri_raw_sequence'], input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'], input_dict['struct'])
            elif self.multimodality_type in ['add']:
                output_dict = self.model(input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'], input_dict['struct'])
            elif self.multimodality_type == 'transfer':
                output_dict = self.model(input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
        else:
            if self.dataset_name in ['fMRI_timeseries', 'hcp']:
                if self.fmri_type in ['timeseries', 'frequency', 'time_domain_low', 'time_domain_ultralow', 'frequency_domain_low', 'frequency_domain_ultralow']:
                    output_dict = self.model(input_dict['fmri_sequence'])
                elif self.fmri_type in ['divided_frequency', 'timeseries_and_frequency']:
                    output_dict = self.model(input_dict['fmri_sequence'], input_dict['fmri_lowfreq_sequence'], input_dict['fmri_ultralowfreq_sequence'])
            elif self.dataset_name == 'DTI':
                ## swinv2
                output_dict = self.model(input_dict['dti'])
            elif self.dataset_name == 'sMRI':
                ## swinv2
                output_dict = self.model(input_dict['smri'])
            elif self.dataset_name == 'DTI+sMRI':
                ## swinv2
                output_dict = self.model(input_dict['struct'])   
            elif self.dataset_name =='struct':
                output_dict = self.model(input_dict['smri'], input_dict['dti'])
            
        torch.cuda.nvtx.range_push("aggregate_losses")
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)
        torch.cuda.nvtx.range_pop()
        #if self.task.lower() ==  'bert' or 'test':
        self.compute_accuracy(input_dict, output_dict)
        return loss_dict, loss
    
    def aggregate_losses(self,input_dict,output_dict):
        final_loss_dict = {}
        final_loss_value = 0
        for loss_name, current_loss_dict in self.writer.losses.items():
            if current_loss_dict['is_active']:
                loss_func = getattr(self, 'compute_' + loss_name)
                torch.cuda.nvtx.range_push(f"{loss_name}")
                current_loss_value = loss_func(input_dict,output_dict)
                torch.cuda.nvtx.range_pop()
                if current_loss_value.isnan().sum() > 0:
                    warnings.warn('found nans in computation')
                    print('at {} loss'.format(loss_name))
                    if self.task.lower() not in ['funcstruct', 'multvit']:
                        self.nan_list+=np.array(input_dict['subject_name'])[(output_dict['reconstructed_fmri_sequence'].reshape(output_dict['reconstructed_fmri_sequence'].shape[0],-1).isnan().sum(axis=1).detach().cpu().numpy() > 0)].tolist()
                    else:
                        self.nan_list+=np.array(input_dict['subject_name'])[(output_dict[self.fine_tune_task].reshape(output_dict[self.fine_tune_task].shape[0],-1).isnan().sum(axis=1).detach().cpu().numpy() > 0)].tolist()
                    print('current_nan_list:',set(self.nan_list))
                
                lamda = current_loss_dict['factor']
                factored_loss = current_loss_value * lamda
                final_loss_dict[loss_name] = factored_loss.item()
                final_loss_value += factored_loss
        final_loss_dict['total'] = final_loss_value.item()
#         with open("conv_nan_sub_list.txt", mode="w") as file:
#             file.write('\n'.join(self.nan_list))
        return final_loss_dict, final_loss_value


        
    def testing(self):
        self.eval_epoch('test')
        self.writer.loss_summary(lr=0)
        self.writer.accuracy_summary(mid_epoch=False)
        for metric_name in dir(self.writer):
            if 'history' not in metric_name:
                continue
            # metric_name =  save history to csv
            metric_score = getattr(self.writer, metric_name)
            #metric score is: [0.7304385997718428]
            #metric_score = metric_score[-1]
            #print('final test score - {} = {}'.format(metric_name,metric_score))
    
    def train(self):
        self.mode = 'train'
        self.model = self.model.train()
        
    def eval(self,set):
        self.mode = set
        self.model = self.model.eval()

    def get_last_loss(self):
        if self.kwargs.get('fine_tune_task') == 'regression': #self.model.task
            return self.writer.val_MAE[-1]
        else:
            return self.writer.total_val_loss_history[-1]

    def get_last_AUROC(self):
        if hasattr(self.writer,'val_AUROC'):
            return self.writer.val_AUROC[-1]
        else:
            return None

    def save_checkpoint_(self, epoch, batch_idx, scaler):
        loss = self.get_last_loss()
        AUROC = self.get_last_AUROC()
        title = str(self.writer.experiment_title) + '_epoch_' + str(int(epoch)) + '_batch_index_'+ str(batch_idx) # ??? ?????? ???????????? ????????? ???~
        
        directory = self.writer.experiment_folder

        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.amp:
            amp_state = scaler.state_dict()

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':self.model.module.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict() if self.optimizer is not None else None,
            'epoch':epoch,
            'loss_value':loss,
            'amp_state': amp_state}
        if AUROC is not None:
            ckpt_dict['AUROC'] = AUROC
        if self.lr_handler.schedule is not None:
            ckpt_dict['schedule_state_dict'] = self.lr_handler.schedule.state_dict()
            ckpt_dict['lr'] = self.optimizer.param_groups[0]['lr']
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path
        
        # Save checkpoint per one epoch 
        # commented out by JB
        # name = "{}.pth".format(core_name) 
        # torch.save(ckpt_dict, os.path.join(directory, name))
        
        core_name = title
        # best loss??? best AUROC??? ?????? ????????? ???????????? ??????
        # classification
        if AUROC is not None and self.best_AUROC < AUROC:
            #self.best_accuracy = accuracy
            self.best_AUROC = AUROC
            #name = "{}_BEST_val_accuracy.pth".format(core_name)
            name = "{}_BEST_val_AUROC.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print(f'updating best saved model with AUROC:{AUROC}')
            #print(f'updating best saved model with accuracy:{accuracy}')
        # regression
        elif AUROC is None and self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST_val_loss.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print(f'updating best saved model with loss: {loss}')


    def compute_reconstruction(self,input_dict,output_dict):
        fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        reconstruction_loss = self.reconstruction_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return reconstruction_loss

    def compute_intensity(self,input_dict,output_dict):
        per_voxel = input_dict['fmri_sequence'][:,1,:,:,:,:]
        torch.cuda.nvtx.range_push("get_intensity_voxels")
        voxels = get_intense_voxels(per_voxel, output_dict['reconstructed_fmri_sequence'].shape, self.gpu) 
        torch.cuda.nvtx.range_pop()
        output_intense = output_dict['reconstructed_fmri_sequence'] * voxels  #[voxels]
        truth_intense = input_dict['fmri_sequence'][:,0] * voxels.squeeze(1) #[voxels.squeeze(1)]
        torch.cuda.nvtx.range_push("self.intensity_loss_func")
        intensity_loss = self.intensity_loss_func(output_intense.squeeze(), truth_intense.squeeze())
        torch.cuda.nvtx.range_pop()
        return intensity_loss

    def compute_perceptual(self,input_dict,output_dict):
        fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        perceptual_loss = self.perceptual_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return perceptual_loss
    
    def compute_contrastive(self,input_dict,output_dict):
        # fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        # print('shape of fmri_sequence is:', fmri_sequence.shape) [batch, channel, width, height, depth, T] [2, 1, 75, 93, 81, 20]
        contrastive_loss = self.contrastive_loss_func(output_dict['transformer_output_sequence'])
        return contrastive_loss
    
    def compute_merge(self,input_dict,output_dict):
        # fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        # print('shape of fmri_sequence is:', fmri_sequence.shape) [batch, channel, width, height, depth, T] [2, 1, 75, 93, 81, 20]
        merge_loss = self.merge_loss_func(output_dict['processed_raw'], output_dict['embedding_per_ROIs'])
        return merge_loss
    
    
    def compute_mask(self, input_dict, output_dict):
        '''
        shape of output of fmri_sequence is: torch.Size([4, 20, 2640])
        '''
        mask_loss = self.mask_loss_func(output_dict['transformer_input_sequence'], output_dict['mask_list'], output_dict['transformer_output_sequence_for_mask_learning'])
        return mask_loss

    def compute_binary_classification(self,input_dict,output_dict):
        binary_loss = self.binary_classification_loss_func(output_dict['binary_classification'].squeeze(), input_dict[self.target].squeeze().float()) # BCEWithLogitsLoss
        #self.binary_classification_loss_func(output_dict['binary_classification'].squeeze(), input_dict['subject_binary_classification'].squeeze())
        
        return binary_loss

    def compute_regression(self,input_dict,output_dict):
        gender_loss = self.regression_loss_func(output_dict['regression'].squeeze(),input_dict[self.target].squeeze()) #self.regression_loss_func(output_dict['regression'].squeeze(),input_dict['subject_regression'].squeeze())
        return gender_loss

    def compute_accuracy(self,input_dict,output_dict):
        task = self.kwargs.get('fine_tune_task') #self.model.task
        #print('in compute accuracy function, task is:', task) # need to be 'binary classification'
        out = output_dict[task].detach().clone().cpu()
        #print(out)
        score = out.squeeze() if out.shape[0] > 1 else out
        labels = input_dict[self.target].clone().cpu() # input_dict['subject_' + task].clone().cpu()
        subjects = input_dict['subject'].clone().cpu()
        for i, subj in enumerate(subjects):
            subject = str(subj.item())
            if subject not in self.writer.subject_accuracy:
                self.writer.subject_accuracy[subject] = {'score': score[i].unsqueeze(0), 'mode': self.mode, 'truth': labels[i],'count': 1}
            else:
                self.writer.subject_accuracy[subject]['score'] = torch.cat([self.writer.subject_accuracy[subject]['score'], score[i].unsqueeze(0)], dim=0)
                self.writer.subject_accuracy[subject]['count'] += 1

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs

import os
from abc import ABC, abstractmethod
import torch
from transformers import BertConfig,BertPreTrainedModel, BertModel
from datetime import datetime
import torch.nn as nn
from nvidia_blocks import *
from modules.crossmodal_transformer import TransformerEncoder
import torch.nn.functional as F
import numpy as np
#import torchaudio.functional as AF


## in case of swin transformer V2
from modules.swin_v2_module import *

## in case of SwinFusion
from modules.swinfusion_module import *

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.best_loss = 1000000
        #self.best_accuracy = 0
        self.best_AUROC = 0
    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def register_vars(self,**kwargs):
        intermediate_vec = kwargs.get('intermediate_vec') # embedding size(h) 
        self.transformer_dropout_rate = kwargs.get('transformer_dropout_rate')

        if kwargs.get('dataset_name') in ['multimodal', 'fMRI_timeseries', 'hcp']:
            if kwargs.get('fmri_type') == 'divided_frequency':
                self.BertConfig = BertConfig(hidden_size=intermediate_vec, vocab_size=1,
                             num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                             num_attention_heads=kwargs.get('num_heads_2DBert'), max_position_embeddings=kwargs.get('sequence_length')+1,
                             hidden_dropout_prob=self.transformer_dropout_rate)
            else:
                self.BertConfig = BertConfig(hidden_size=intermediate_vec, vocab_size=1,
                             num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                             num_attention_heads=kwargs.get('num_heads_2DBert'), max_position_embeddings=kwargs.get('sequence_length')+1,
                             hidden_dropout_prob=self.transformer_dropout_rate)#, torchscript=True)
        elif kwargs.get('dataset_name') == 'fMRI_image':
            self.BertConfig = BertConfig(hidden_size=intermediate_vec, vocab_size=1,
                                         num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                         num_attention_heads=16, max_position_embeddings=30,
                                         hidden_dropout_prob=self.transformer_dropout_rate)

        elif kwargs.get('dataset_name') in ['DTI', 'sMRI']:
            vit_max_pos_embed = (intermediate_vec // kwargs.get('patch_size')) ** 2 # 441
            self.BertConfig = BertConfig(hidden_size=intermediate_vec, vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=kwargs.get('num_heads_2DBert'), max_position_embeddings=vit_max_pos_embed+1,
                                     hidden_dropout_prob=self.transformer_dropout_rate) # 그냥..
        self.label_num = 1
        #self.inChannels = 2
        #self.outChannels = 1
        self.model_depth = 4
        self.intermediate_vec = intermediate_vec
        self.use_cuda = kwargs.get('gpu') #'cuda'
        self.shapes = kwargs.get('shapes')
        self.dataset_name = kwargs.get('dataset_name')
        self.feature_map_size = kwargs.get('feature_map_size')

    def load_partial_state_dict(self, state_dict,load_cls_embedding):
        print('loading parameters onto new model...')
        own_state = self.state_dict()
        loaded = {name:False for name in own_state.keys()}
        for name, param in state_dict.items():
            if name not in own_state:
                print('notice: {} is not part of new model and was not loaded.'.format(name))
                continue
            elif 'cls_embedding' in name and not load_cls_embedding:
                continue
            elif 'position' in name and param.shape != own_state[name].shape:
                print('debug line above')
                continue
            param = param.data
            own_state[name].copy_(param)
            loaded[name] = True
        for name,was_loaded in loaded.items():
            if not was_loaded:
                print('notice: named parameter - {} is randomly initialized'.format(name))

    # last epoch만 저장하는 코드
    def save_checkpoint(self, directory, title, epoch, loss, AUROC, optimizer=None,schedule=None):
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':self.state_dict(),
            'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,
            'epoch':epoch,
            'loss_value':loss}
        if AUROC is not None:
            ckpt_dict['AUROC'] = AUROC
        if schedule is not None:
            ckpt_dict['schedule_state_dict'] = schedule.state_dict()
            ckpt_dict['lr'] = schedule.get_last_lr()[0]
        if hasattr(self,'loaded_model_weights_path'):
            ckpt_dict['loaded_model_weights_path'] = self.loaded_model_weights_path
        
        # Save checkpoint per one epoch - 아직 one epoch도 못 돌았음. 이거 하는 거 의미 없음^%^
        # core_name = title
        # print('saving ckpt of {}_epoch'.format(epoch))
        # name = "{}_epoch_{}.pth".format(core_name, epoch)
        # torch.save(ckpt_dict, os.path.join(directory, name))
        
        # Save the file with specific name
        core_name = title
        name = "{}_last_epoch.pth".format(core_name)
        torch.save(ckpt_dict, os.path.join(directory, name)) 
        
        # best loss나 best AUROC를 가진 모델만 저장하는 코드
        if AUROC is None and self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST_val_loss.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print('updating best saved model...')
        if AUROC is not None and self.best_AUROC < AUROC:
            self.best_AUROC = AUROC
            name = "{}_BEST_val_AUROC.pth".format(core_name)
            torch.save(ckpt_dict, os.path.join(directory, name))
            print('updating best saved model...')

    
class Transformer_Block(BertPreTrainedModel, BaseModel):
    def __init__(self,config,**kwargs):
        super(Transformer_Block, self).__init__(config)
        self.register_vars(**kwargs)
        self.cls_pooling = True
        self.bert = BertModel(self.BertConfig, add_pooling_layer=self.cls_pooling)
        self.init_weights()
        self.register_buffer('cls_id', (torch.ones((1, 1, self.BertConfig.hidden_size)) * 0.5), persistent=False)
        self.cls_embedding = nn.Sequential(nn.Linear(self.BertConfig.hidden_size, self.BertConfig.hidden_size), nn.LeakyReLU())
        
    def concatenate_cls(self, x):
        # x size : (1, 361, 48) (batch_size, sequence_length, hidden_size)
        # cls id size : (1, 1, 48) 그래도 8으로 늘릴 수는 있잖아~
        cls_token = self.cls_embedding(self.cls_id.expand(x.size()[0], -1, -1))
        # cls token size : (1, 1, 48)
        
        return torch.cat([cls_token, x], dim=1)


    def forward(self, x):
        inputs_embeds = self.concatenate_cls(x) # (8, 362, 48)
        outputs = self.bert(input_ids=None,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=inputs_embeds, #give our embeddings
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=self.BertConfig.use_return_dict
                            ) #여기서 지정해줘야 함.

        
        sequence_output = outputs[0][:, 1:, :] # torch.Size([8, 361, 84])
        pooled_cls = outputs[1] # torch.Size([8, 84])

        return {'sequence': sequence_output, 'cls': pooled_cls}

class Transformer_Net(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Net, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.feature_squeeze = kwargs.get('feature_squeeze')
        self.register_vars(**kwargs)
        if self.feature_squeeze == True:
            if self.feature_map_gen == 'convolution_ul':
                self.proj_u = nn.Conv1d(368, 128, kernel_size=1, padding=0, bias=False)
        # transformer
        self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        if self.task == 'regression':
            self.final_activation_func = nn.LeakyReLU()
        elif self.task == 'binary_classification':
            self.final_activation_func = nn.Sigmoid()
            self.label_num = 1
        self.regression_head = nn.Linear(self.intermediate_vec, self.label_num) #.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        
        if self.dataset_name == 'fMRI_image':
            batch_size, inChannels, W, H, D, T = x.shape
            x = x.reshape(batch_size, inChannels, W*H*D, T) # (batch_size, 984555, 373) 이어야 함. 
            x = x.contiguous(memory_format=torch.channels_last_3d) # changed from NCHDW to NHWDC format for accellerating
        elif self.dataset_name == 'fMRI_timeseries':
            batch_size, T, ROIs = x.shape # (batch size, 368, 84)

        torch.cuda.nvtx.range_push("transformer")
        if self.feature_squeeze == True:
            if self.feature_map_gen == 'convolution_ul':
                x = self.proj_u(x) # torch.Size([1, 184, 48])
        
        
        transformer_dict = self.transformer(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("reshape")
        
        '''
        size of out seq is: torch.Size([1, 361, 84])
        size of out cls is: torch.Size([1, 84])
        size of prediction is: torch.Size([1, 1])
        '''
        out_seq = transformer_dict['sequence']
        out_cls = transformer_dict['cls']
        prediction = self.regression_head(out_cls)
        return {'reconstructed_fmri_sequence': out_seq, 'embedding_per_ROIs': out_cls, self.task:prediction}

class Transformer_Net_Two_Channels(BaseModel):
    def __init__(self, **kwargs):
        super(Transformer_Net_Two_Channels, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)
        self.concat_method = kwargs.get('concat_method')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.sequence_length = kwargs.get('sequence_length')
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.use_merge_loss = kwargs.get('use_merge_loss')
        # transformer - 일단 seq len이 둘 다 368인 걸 기준으로 짰음. 나중에 또 바꿀 거임.
        # why 128? 368//3과 가장 가까운 16의 배수이기 때문.
        if self.use_merge_loss == True:
            self.transformer_raw = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        
        if self.feature_map_size == 'same':
            self.transformer_low = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
            self.transformer_ultralow = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        elif self.feature_map_size == 'different':
            self.BertConfig_ultralow = BertConfig(hidden_size=self.intermediate_vec, vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=kwargs.get('num_heads_2DBert'), max_position_embeddings=128+1,
                                     hidden_dropout_prob=0.2) 
            self.transformer_ultralow = Transformer_Block(self.BertConfig_ultralow, **kwargs).to(memory_format=torch.channels_last_3d)
            self.transformer_low = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

            if self.feature_map_gen == 'convolution_ul':
                self.proj_u = nn.Conv1d(self.sequence_length, 128, kernel_size=1, padding=0, bias=False)
                
        if self.concat_method == 'concat':
            self.proj_layer = nn.Linear(2*self.intermediate_vec, self.intermediate_vec) 
        self.regression_head = nn.Linear(self.intermediate_vec, self.label_num) #.to(memory_format=torch.channels_last_3d)

    def forward(self, x, x_l, x_u):
        
#         if self.dataset_name == 'fMRI_image':
#             batch_size, inChannels, W, H, D, T = x.shape
#             x = x.reshape(batch_size, inChannels, W*H*D, T) # (batch_size, 984555, 373) 이어야 함. 
#             x = x.contiguous(memory_format=torch.channels_last_3d) # changed from NCHDW to NHWDC format for accellerating
#             print('shape of x after permutation:', x.size()) 
#         elif self.dataset_name == 'fMRI_timeseries':
#             batch_size, T, ROIs = x_l.shape # (batch size, 368, 84)

        torch.cuda.nvtx.range_push("transformer")
        if self.use_merge_loss == True:
            transformer_dict_raw = self.transformer_raw(x)
            out_cls_raw = transformer_dict_raw['cls'] # torch.Size([1, 84])
        
        if self.feature_map_size == 'same':
            transformer_dict_low = self.transformer_low(x_l)
            transformer_dict_ultralow = self.transformer_ultralow(x_u)
            
        elif self.feature_map_size == 'different':
            if self.feature_map_gen == 'convolution_ul':
                x_u = self.proj_u(x_u) # torch.Size([1, 128, 84])
            transformer_dict_low = self.transformer_low(x_l)
            transformer_dict_ultralow = self.transformer_ultralow(x_u)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("reshape")
        
        '''
        size of out seq is: torch.Size([1, 361, 84])
        size of out cls is: torch.Size([1, 84])
        size of prediction is: torch.Size([1, 1])
        '''
        
        out_cls_low = transformer_dict_low['cls'] # torch.Size([1, 84])
        out_cls_ultralow = transformer_dict_ultralow['cls'] # torch.Size([1, 84])
        
        if self.concat_method == 'concat':
            out_cls = torch.cat([out_cls_low, out_cls_ultralow], dim=1) # (1, 84*2)
            out_cls = self.proj_layer(out_cls) # (1, 84)
        elif self.concat_method == 'hadamard':
            out_cls = torch.mul(out_cls_low, out_cls_ultralow) # (1, 84)
        
        # out_cls랑 out_cls_raw가 두 가지가 비슷하도록 loss를 짜야 함!
    
        prediction = self.regression_head(out_cls) # torch.Size([1, 1])
        
        if self.use_merge_loss == True:
            ans_dict = {'processed_raw': out_cls_raw, 'embedding_per_ROIs': out_cls, self.task:prediction}
        else:
            ans_dict = {'embedding_per_ROIs': out_cls, self.task:prediction}
        return ans_dict
    
class MULTModel(BaseModel):
    def __init__(self, **kwargs):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        ## size of features : 48 
        self.orig_d_l, self.orig_d_u = kwargs.get('intermediate_vec'), kwargs.get('intermediate_vec')
        
        ## 그냥 embedding dimension.. 굳이 48일 필요는 없으나 난 필요한 게 있기 때문에..
        self.d_l, self.d_u = kwargs.get('intermediate_vec'), kwargs.get('intermediate_vec')
        
        self.num_heads_mult = kwargs.get('num_heads_mult')
        self.layers = kwargs.get('nlevels')
        self.attn_dropout = kwargs.get('attn_dropout')
        self.attn_dropout_u = kwargs.get('attn_dropout_u')
        self.relu_dropout = kwargs.get('relu_dropout')
        self.res_dropout = kwargs.get('res_dropout')
        self.out_dropout = kwargs.get('out_dropout')
        self.embed_dropout = kwargs.get('embed_dropout')
        self.attn_mask = kwargs.get('attn_mask')
        self.sequence_length = kwargs.get('sequence_length')
        self.fine_tune_task = kwargs.get('fine_tune_task')
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.mixing = kwargs.get('mixing')
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.concat_method = kwargs.get('concat_method')
        self.fmri_type = kwargs.get('fmri_type')
        combined_dim = self.d_l + self.d_u  # 96
        
        if self.fine_tune_task == 'binary_classification':
            output_dim = 1 # regression이면..  뭐..미래의 내가 알아서..
        # 1. Temporal Convolutional Layers
        if self.feature_map_size == 'different':
            if self.feature_map_gen == 'convolution_ul+l':
                self.proj_l = nn.Conv1d(self.sequence_length, self.sequence_length, kernel_size=1, padding=0, bias=False)
                self.proj_u = nn.Conv1d(self.sequence_length, self.sequence_length//2, kernel_size=1, padding=0, bias=False)
            elif self.feature_map_gen == 'convolution_ul':
                #self.proj_l = nn.Conv1d(self.sequence_length, self.sequence_length, kernel_size=1, padding=0, bias=False)
                self.proj_u = nn.Conv1d(self.sequence_length, self.sequence_length//2, kernel_size=1, padding=0, bias=False)
           
        elif self.feature_map_size == 'same':
            if self.feature_map_gen == 'convolution_ul+l':
                self.proj_l = nn.Conv1d(self.sequence_length, self.sequence_length, kernel_size=1, padding=0, bias=False)
        
        # 2. Crossmodal Attentions
        self.trans_l_with_u = self.get_network(self_type='lu') # low to ultra low
        self.trans_u_with_l = self.get_network(self_type='ul') # ultra low to low
        
        ## if feature map size different,
        if self.feature_map_size == 'different':
            self.deconv = nn.ConvTranspose1d(self.sequence_length//2, self.sequence_length, kernel_size=1, padding=0, bias=False)
        elif self.fmri_type == 'timeseries_and_frequency':
            self.deconv = nn.ConvTranspose1d(self.sequence_length//2, self.sequence_length, kernel_size=1, padding=0, bias=False)
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_mem = self.get_network(self_type='mem', layers=3) # self attention
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_u_mem = self.get_network(self_type='u_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer1 = nn.Linear(combined_dim, combined_dim//2)
        self.out_layer2 = nn.Linear(combined_dim//2, output_dim)

    def get_network(self, self_type='lu', layers=-1):
        if self.feature_map_size == 'same':
            if self_type in ['ul']:
                embed_dim, attn_dropout = self.d_l, self.attn_dropout
            elif self_type in ['lu']:
                embed_dim, attn_dropout = self.d_u, self.attn_dropout_u
            elif self_type == 'l_mem':
                embed_dim, attn_dropout = self.d_l, self.attn_dropout
            elif self_type == 'u_mem':
                embed_dim, attn_dropout = self.d_u, self.attn_dropout_u
            elif self_type == 'mem':
                embed_dim, attn_dropout = self.d_u*2, self.attn_dropout
            else:
                raise ValueError("Unknown network type")
        elif self.feature_map_size == 'different':
            if self_type in ['ul']:
                embed_dim, attn_dropout = self.d_l, self.attn_dropout
            elif self_type in ['lu']:
                embed_dim, attn_dropout = self.d_u, self.attn_dropout_u
            elif self_type == 'l_mem':
                embed_dim, attn_dropout = self.d_l, self.attn_dropout
            elif self_type == 'u_mem':
                embed_dim, attn_dropout = self.d_u, self.attn_dropout_u
            elif self_type == 'mem':
                embed_dim, attn_dropout = self.d_u*2, self.attn_dropout
            else:
                raise ValueError("Unknown network type")
        # embed dim -> 368 / 736
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads_mult=self.num_heads_mult,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_l, x_u):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features] - (1, 368, 48)
        T : 368 -> d_l, d_u로 assign
        D : 48 -> orig_d_l, orig_d_u로 assign
        """
        if self.feature_map_size == 'different':
            '''
            ultra low should have smaller feature map. so compress input data!
            '''
            if self.feature_map_gen == 'convolution_ul+l':
                proj_x_l = self.proj_l(F.dropout(x_l, p=self.embed_dropout, training=self.training)) # torch.Size([1, 368, 48])
                proj_x_l = proj_x_l.transpose(1, 2) #torch.Size([1, 48, 368])

                proj_x_u = self.proj_u(x_u) # torch.Size([1, 184, 48])
                proj_x_u = proj_x_u.transpose(1, 2) #torch.Size([1, 48, 184])
            elif self.feature_map_gen == 'convolution_ul':
                # no conv for low frequency
                proj_x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training) # torch.Size([1, 48, 368])
                
                # yes conv for ultra low frequency
                proj_x_u = self.proj_u(x_u)
                proj_x_u = proj_x_u.transpose(1, 2) #torch.Size([1, 48, 184])

        elif self.feature_map_size == 'same':
            if self.feature_map_gen == 'no':
                proj_x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training) # torch.Size([1, 48, 368])
                proj_x_u = x_u.transpose(1, 2)# torch.Size([1, 48, 368])
            elif self.feature_map_gen == 'convolution_ul+l':
                proj_x_l = self.proj_l(F.dropout(x_l, p=self.embed_dropout, training=self.training)) # torch.Size([1, 368, 48])
                proj_x_l = proj_x_l.transpose(1, 2) #torch.Size([1, 48, 368])

                proj_x_u = self.proj_l(x_u) # torch.Size([1, 368, 48])
                proj_x_u = proj_x_u.transpose(1, 2) #torch.Size([1, 48, 368])
        
        # Project the textual/visual/audio features: 이게 crossmodal transformer의 input - 여기서!! 여기서 들어갈 때!! 여기가 feature map임!
        proj_x_u = proj_x_u.permute(2, 0, 1) #torch.Size([368, 1, 48])
        proj_x_l = proj_x_l.permute(2, 0, 1) #torch.Size([368, 1, 48])
        
        # Cross attention Transformer (transformer encoder가 여기서 일을 시작함!)
        # U --> L & L --> U
        h_l_with_us = self.trans_l_with_u(proj_x_l, proj_x_u, proj_x_u) #torch.Size([368, 1, 48]) - 여기서 k q v 교환
        h_u_with_ls = self.trans_u_with_l(proj_x_u, proj_x_l, proj_x_l) #torch.Size([368, 1, 48])
        if self.feature_map_size == 'different':
            h_u_with_ls = self.deconv(h_u_with_ls.transpose(0, 1).float()) # (1, 184, 48) -> (1, 368, 48)
            h_u_with_ls = h_u_with_ls.transpose(0, 1) # (368, 1, 48)
        elif self.fmri_type == 'timeseries_and_frequency':
            h_u_with_ls = self.deconv(h_u_with_ls.transpose(0, 1).float()) # (1, 184, 48) -> (1, 368, 48)
            h_u_with_ls = h_u_with_ls.transpose(0, 1) # (368, 1, 48)
        '''
        if feature map size different,
        shape of h_l_with_us is: torch.Size([368, 8, 48])
        shape of h_u_with_ls is: torch.Size([184, 8, 48])
        '''
        
        
        if self.mixing == 'U2L_and_L2U':
            if self.concat_method == 'concat':
                h_zs = torch.cat([h_l_with_us, h_u_with_ls], dim=2) #torch.Size([368, 1, 96] (이 둘은 똑같아야 함)

                # Transformer with self attention
                h_zs = self.trans_mem(h_zs) # torch.Size([368, 1, 96])
                if type(h_zs) == tuple:
                    h_zs = h_zs[0]
                last_h_z = last_hs = h_zs[-1]   # Take the last output for prediction  - torch.Size([1, 96])
                out_cls = self.out_layer1(last_h_z) # torch.Size([1, 48])
            elif self.concat_method == 'hadamard':
#                 h_zs = torch.empty(h_l_with_us.shape).to(h_l_with_us.device)
#                 for i in range(h_l_with_us.shape[0]):
#                     h_zs_ = torch.mul(h_l_with_us[i, :, :], h_u_with_ls[i, :, :]) # torch.Size([1, 1, 48]
#                     h_zs[i, :, :] = h_zs_
                h_zs = torch.mul(h_l_with_us, h_u_with_ls) # torch.Size([368, 1, 48] 
                
                # Transformer with self attention
                h_zs = self.trans_l_mem(h_zs) # torch.Size([368, 1, 48])
                if type(h_zs) == tuple:
                    h_zs = h_zs[0]
                out_cls = last_hs=h_zs[-1] # Take the last output for prediction - torch.Size([1, 48])
        
        elif self.mixing == 'U2L':
            # Transformer with self attention
            h_zs = self.trans_l_mem(h_l_with_us) # torch.Size([368, 1, 48])
            if type(h_zs) == tuple:
                h_zs = h_zs[0]
            out_cls = last_hs = h_zs[-1]   # Take the last output for prediction - torch.Size([1, 48])
            
        elif self.mixing == 'L2U':
            # Transformer with self attention
            h_zs = self.trans_u_mem(h_u_with_ls) # torch.Size([368, 1, 48])
            if type(h_zs) == tuple:
                h_zs = h_zs[0]
            out_cls = last_hs = h_zs[-1]   # Take the last output for prediction - torch.Size([1, 48])  
        
        pred = self.out_layer2(out_cls) # torch.Size([1, 1])

        '''
        # A residual block.. 아니면 structure랑 structure에 올라간 fMRI를 이런 식으로 residual block으로 만드는 방법도 생각해봐야 할 듯?
        print('self.proj1 is:', self.proj1)
        print('self.proj2 is:', self.proj2)
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h_z)), p=self.out_dropout, training=self.training))
        
        print('shape of last_hs_proj is:', last_hs_proj.shape)
        last_hs_proj += last_hs
        
        out_cls = self.out_layer1(last_hs_proj)
        pred = self.out_layer1(out_cls) #torch.Size([1, 1])
        '''
        
        return {'embedding_per_ROIs': out_cls, self.task:pred}

# class VIT(BaseModel):
#     def __init__ (self, **kwargs):
#         """
#         Construct a VIT model
#         """
#         super(VIT, self).__init__()
#         self.task = kwargs.get('fine_tune_task')
#         self.num_heads_mult = kwargs.get('num_heads_mult')
#         self.layers = kwargs.get('nlevels')
#         self.attn_dropout = kwargs.get('attn_dropout')
#         self.relu_dropout = kwargs.get('relu_dropout')
#         self.res_dropout = kwargs.get('res_dropout')
#         self.out_dropout = kwargs.get('out_dropout')
#         self.embed_dropout = kwargs.get('embed_dropout')
#         self.attn_mask = kwargs.get('attn_mask')
#         self.fine_tune_task = kwargs.get('fine_tune_task')
#         self.patch_size = kwargs.get('patch_size')
#         self.intermediate_vec = kwargs.get('intermediate_vec')
        
#         self.register_vars(**kwargs)
        
#         # make patch
#         self.patch_embed = nn.Conv2d(1, self.intermediate_vec, kernel_size=self.patch_size, stride=self.patch_size)
        
        
#         # make transformer
        
#         self.transformer = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        
#         self.regression_head = nn.Linear(self.intermediate_vec, self.label_num)

#     def forward(self, x):
#         """
#         (batch_size, number of patches, embedding)
#         text, audio, and vision should have dimension [batch_size, seq_len, n_features]
#         dti : (1, 84, 84) -> (1, 441, embed) ; 4 * 4 patch가 441개~
#         smri : (1, 84, 84) -> (1, 441, embed) ; 4 * 4 patch가 441개~ # 얘 되게 sparse한데.. 되겠지...?
#         """
#         # divide into patches
#         x = x.unsqueeze(dim=1) # B, C, H, W (1, 1, 84 ,84)
#         #B, C, H, W = x.shape
#         x = self.patch_embed(x).flatten(2).transpose(1, 2)  # B Ph*Pw C (1, 441, 24) # 24 is embedding size
    
#         # run transformer
#         transformer_dict = self.transformer(x)
#         out_seq = transformer_dict['sequence']
#         out_cls = transformer_dict['cls']
#         prediction = self.regression_head(out_cls)
#         return {self.task : prediction}
    
    
    
# class MultVIT(BaseModel):
#     def __init__(self, **kwargs):
#         """
#         Construct a MulTViT model.
#         dti (84, 84) -> smri (84, 84)
#         smri (84, 84) -> dti (84, 84)
#         """
#         super(MultVIT, self).__init__()
#         self.task = kwargs.get('fine_tune_task')
#         ## size of features : 84 
#         self.orig_d_s, self.orig_d_d = kwargs.get('intermediate_vec'), kwargs.get('intermediate_vec')
        
#         ## 그냥 embedding dimension.. 굳이 84일 필요는 없으나 난 필요한 게 있기 때문에..
#         self.d_s, self.d_d = kwargs.get('intermediate_vec'), kwargs.get('intermediate_vec')
        
#         self.num_heads_mult = kwargs.get('num_heads_mult')
#         self.layers = kwargs.get('nlevels')
#         self.attn_dropout = kwargs.get('attn_dropout')
#         self.relu_dropout = kwargs.get('relu_dropout')
#         self.res_dropout = kwargs.get('res_dropout')
#         self.out_dropout = kwargs.get('out_dropout')
#         self.embed_dropout = kwargs.get('embed_dropout')
#         self.attn_mask = kwargs.get('attn_mask')
#         self.fine_tune_task = kwargs.get('fine_tune_task')
#         self.feature_map_gen = kwargs.get('feature_map_gen')
#         self.feature_map_size = kwargs.get('feature_map_size')
#         self.mixing = kwargs.get('mixing')
#         self.feature_map_gen = kwargs.get('feature_map_gen')
#         self.concat_method = kwargs.get('concat_method')
#         self.patch_size = kwargs.get('patch_size')
#         self.embed_dim_mult = kwargs.get('embed_dim_mult')
#         combined_dim = 2 * self.embed_dim_mult 
        
#         if self.fine_tune_task == 'binary_classification':
#             output_dim = 1
#         # 1. Temporal Convolutional Layers
# #         if self.feature_map_size == 'different':
# #             if self.feature_map_gen == 'convolution_ul+l':
# #                 self.proj_l = nn.Conv1d(self.sequence_length, self.sequence_length, kernel_size=1, padding=0, bias=False)
# #                 self.proj_u = nn.Conv1d(self.sequence_length, self.sequence_length//2, kernel_size=1, padding=0, bias=False)
# #             elif self.feature_map_gen == 'convolution_ul':
# #                 #self.proj_l = nn.Conv1d(self.sequence_length, self.sequence_length, kernel_size=1, padding=0, bias=False)
# #                 self.proj_u = nn.Conv1d(self.sequence_length, self.sequence_length//2, kernel_size=1, padding=0, bias=False)
           
# #         elif self.feature_map_size == 'same':
# #             if self.feature_map_gen == 'convolution_ul+l':
# #                 self.proj_l = nn.Conv1d(self.sequence_length, self.sequence_length, kernel_size=1, padding=0, bias=False)
        
#         # 2. Crossmodal Attentions
#         self.trans_d_with_s = self.get_network(self_type='ds') # low to ultra low
#         self.trans_s_with_d = self.get_network(self_type='sd') # ultra low to low
        
#         ## if feature map size different,
#         # embed_size1 = 84//int(self.patch_size) # for dti & smri
#         # embed_size2 = 4//int(self.patch_size) # for smri
#         # embed_dti = int(embed_size1 ** 2)
#         # embed_smri = int(embed_size1*embed_size2)
#         # self.deconv =  nn.ConvTranspose1d(embed_smri, embed_dti, kernel_size=1, padding=0, bias=False)
        
#         # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
#         #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
#         self.trans_mem = self.get_network(self_type='mem', layers=3) # self attention
#         self.trans_d_mem = self.get_network(self_type='d_mem', layers=3)
#         self.trans_s_mem = self.get_network(self_type='s_mem', layers=3)
       
#         # 4. make patch
#         self.patch_embed_dti = nn.Conv2d(1, self.embed_dim_mult, kernel_size=self.patch_size, stride=self.patch_size)
#         self.patch_embed_smri = nn.Conv2d(1, self.embed_dim_mult, kernel_size=self.patch_size, stride=self.patch_size)
        
#         # Projection layers
#         self.proj1 = nn.Linear(combined_dim, combined_dim)
#         self.proj2 = nn.Linear(combined_dim, combined_dim)
#         self.out_layer1 = nn.Linear(combined_dim, combined_dim//2)
#         self.out_layer2 = nn.Linear(combined_dim//2, output_dim)

#     def get_network(self, self_type='ds', layers=-1):
#         if self_type in ['ds']:
#             embed_dim, attn_dropout = self.embed_dim_mult, self.attn_dropout
#         elif self_type in ['sd']:
#             embed_dim, attn_dropout = self.embed_dim_mult, self.attn_dropout
#         elif self_type == 's_mem':
#             embed_dim, attn_dropout = self.d_s, self.attn_dropout
#         elif self_type == 'd_mem':
#             embed_dim, attn_dropout = self.d_d, self.attn_dropout
#         elif self_type == 'mem':
#             embed_dim, attn_dropout = 2*self.embed_dim_mult, self.attn_dropout
#         else:
#             raise ValueError("Unknown network type")

#         # embed dim -> 368 / 736
#         return TransformerEncoder(embed_dim=embed_dim,
#                                   num_heads_mult=self.num_heads_mult,
#                                   layers=max(self.layers, layers),
#                                   attn_dropout=attn_dropout,
#                                   relu_dropout=self.relu_dropout,
#                                   res_dropout=self.res_dropout,
#                                   embed_dropout=self.embed_dropout,
#                                   attn_mask=self.attn_mask)
            
#     def forward(self, x_s, x_d):
#         """
#         (batch_size, number of patches, embedding)
#         text, audio, and vision should have dimension [batch_size, seq_len, n_features]
#         dti : (1, 84, 84) -> (1, 441, embed) ; 4 * 4 patch가 441개~
#         smri : (1, 84, 84) -> (1, 441, embed) ; 4 * 4 patch가 441개~ # 얘 되게 sparse한데.. 되겠지...?
#         T : 368 -> d_l, d_u로 assign
#         D : 48 -> orig_d_l, orig_d_u로 assign
#         """
#         # divide into patches
#         x_d = x_d.unsqueeze(dim=1) # B, C, H, W (1, 1, 84 ,84)
#         B, C, H, W = x_d.shape
#         x_d = self.patch_embed_dti(x_d).flatten(2).transpose(1, 2)  # B Ph*Pw C (1, 441, 24) # 24 is embedding size
        
#         x_s = x_s.unsqueeze(dim=1) # B, C, H, W (1, 1, 84 ,4)
#         B, C, H, W = x_s.shape
#         x_s = self.patch_embed_smri(x_s).flatten(2).transpose(1, 2)  # B Ph*Pw C (1, 441, 24) # 24 is embedding size
        
#         proj_x_s = F.dropout(x_s.transpose(1, 2), p=self.embed_dropout, training=self.training) # torch.Size([1, 24, 441])
#         proj_x_d = x_d.transpose(1, 2)# torch.Size([1, 24, 441])

        
#         # Project the textual/visual/audio features: 이게 crossmodal transformer의 input - 여기서!! 여기서 들어갈 때!! 여기가 feature map임!
#         proj_x_s = proj_x_s.permute(2, 0, 1) #torch.Size([441, 1, 24])
#         proj_x_d = proj_x_d.permute(2, 0, 1) #torch.Size([441, 1, 24])
        
#         # Cross attention Transformer (transformer encoder가 여기서 일을 시작함!)
#         # s --> d & d --> s
#         h_s_with_ds = self.trans_s_with_d(proj_x_s, proj_x_d, proj_x_d) #torch.Size([441, 1, 24])- 여기서 k q v 교환
#         h_d_with_ss = self.trans_d_with_s(proj_x_d, proj_x_s, proj_x_s) #torch.Size([441, 1, 24])

        
#         #h_s_with_ds = self.deconv(h_s_with_ds.transpose(0, 1).float()) # torch.Size([1, 441, 24))
#         #h_s_with_ds = h_s_with_ds.transpose(0, 1) #torch.Size([441, 1, 24])
#         if self.concat_method == 'concat':
#             h_zs = torch.cat([h_s_with_ds, h_d_with_ss], dim=2) # torch.Size([441, 1, 84*2]) (이 둘은 똑같아야 함)
#             # Transformer with self attention
#             h_zs = self.trans_mem(h_zs) # torch.Size([441, 1, 84*2])
#             if type(h_zs) == tuple:
#                 h_zs = h_zs[0]
#             last_h_z = last_hs = h_zs[-1]   # Take the last output for prediction  - torch.Size([1, 84*2])
#             out_cls = self.out_layer1(last_h_z) # torch.Size([1, 84])
#         # hadamard는 나중에 또 고쳐..
#         elif self.concat_method == 'hadamard':
#             h_zs = torch.mul(h_l_with_us, h_u_with_ls) # torch.Size([368, 1, 48] 

#             # Transformer with self attention
#             h_zs = self.trans_l_mem(h_zs) # torch.Size([368, 1, 48])
#             if type(h_zs) == tuple:
#                 h_zs = h_zs[0]
#             out_cls = last_hs=h_zs[-1] # Take the last output for prediction - torch.Size([1, 48])
        
#         pred = self.out_layer2(out_cls) # torch.Size([1, 1])
        
#         return {'embedding_per_ROIs': out_cls, self.task:pred}    
    
class SwinTransformerV2(BaseModel):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224 -> 84
        patch_size (int | tuple(int)): Patch size. Default: 4 -> 6 -> 7
        in_chans (int): Number of input image channels. Default: 3 -> 1
        num_classes (int): Number of classes for classification head. Default: 1000 -> 1
        embed_dim (int): Patch embedding dimension. Default: 96 -> 12
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads_swin (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7 -> 6
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size_w=84, img_size_h=84, patch_size=7, in_chans=1, num_classes=1,
                 embed_dim=12, depths=[2, 2, 6], num_heads_swin=[3, 6, 12],
                 window_size=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        
        self.task = kwargs.get('fine_tune_task')
        self.num_classes = num_classes
        self.num_layers = len(depths) # 4
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.drop_rate = kwargs.get('drop_rate')
        self.attn_drop_rate = kwargs.get('attn_drop_rate')

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size_w=img_size_w, img_size_h=img_size_h, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches # 441
        patches_resolution = self.patch_embed.patches_resolution  # (21, 21)
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        ## 3번 돌아
        for i_layer in range(self.num_layers):
            input_resolution=(patches_resolution[0] // (2 ** i_layer),
                              patches_resolution[1] // (2 ** i_layer)),
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads_swin=num_heads_swin[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        ## (48, 1) -> 아직 act func 돌기 전..!

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = x.unsqueeze(dim=1) # DTI shape: Batch, 1, 84, 84, on the gpu
        x = self.forward_features(x)
        x = self.head(x)
        #return x
        return {self.task:x}

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

class SwinFusion(BaseModel):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=84, patch_size=7, in_chans=1,
                 embed_dim=12, Ex_depths=[6, 6], Fusion_depths=[2, 2, 2], Re_depths=[6,6], 
                 Ex_num_heads=[6, 6], Fusion_num_heads=[6, 6, 6], Re_num_heads=[6, 6],
                 window_size=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.8, attn_drop_rate=0.8, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinFusion, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        print('in_chans: ', in_chans)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        ## for extracted image ##
        self.swin = SwinTransformerV2(img_size_w=84, img_size_h=84, patch_size=7, in_chans=1, num_classes=1,
                 embed_dim=12, depths=[2, 2, 6], num_heads_swin=[3, 6, 12],
                 window_size=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs
        )
        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        ####修改shallow feature extraction 网络, 修改为2个3x3的卷积####
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim_temp, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.Ex_num_layers = len(Ex_depths)
        self.Fusion_num_layers = len(Fusion_depths)
        self.Re_num_layers = len(Re_depths)

        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed_fusion(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.softmax = nn.Softmax(dim=0)
        # absolute position embedding
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr_Ex = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Ex_depths))]  # stochastic depth decay rule
        dpr_Fusion = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Fusion_depths))]  # stochastic depth decay rule
        dpr_Re = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Re_depths))]  # stochastic depth decay rule
        # build Residual Swin Transformer blocks (RSTB)
        self.layers_Ex_A = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Ex_depths[i_layer],
                         num_heads=Ex_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Ex_A.append(layer)
        self.norm_Ex_A = norm_layer(self.num_features)

        self.layers_Ex_B = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Ex_depths[i_layer],
                         num_heads=Ex_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Ex_B.append(layer)
        self.norm_Ex_B = norm_layer(self.num_features)
        
        self.layers_Fusion = nn.ModuleList()
        for i_layer in range(self.Fusion_num_layers):
            layer = CRSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Fusion_depths[i_layer],
                         num_heads=Fusion_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Fusion[sum(Fusion_depths[:i_layer]):sum(Fusion_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Fusion.append(layer)
        self.norm_Fusion_A = norm_layer(self.num_features)
        self.norm_Fusion_B = norm_layer(self.num_features)
        
        self.layers_Re = nn.ModuleList()
        for i_layer in range(self.Re_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Re_depths[i_layer],
                         num_heads=Re_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Re[sum(Re_depths[:i_layer]):sum(Re_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Re.append(layer)
        self.norm_Re = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body_Ex_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_Ex_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_Fusion = nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_Re = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
            self.conv_last2 = nn.Conv2d(embed_dim_temp, int(embed_dim_temp/2), 3, 1, 1)
            self.conv_last3 = nn.Conv2d(int(embed_dim_temp/2), num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features_Ex_A(self, x):
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))           
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Ex_A:
            x = layer(x, x_size)

        x = self.norm_Ex_A(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features_Ex_B(self, x):    
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))      
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Ex_B:
            x = layer(x, x_size)

        x = self.norm_Ex_B(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features_Fusion(self, x, y):
        input_x = x
        input_y = y        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        
        for layer in self.layers_Fusion:
            x, y = layer(x, y, x_size)
            # y = layer(y, x, x_size)
            

        x = self.norm_Fusion_A(x)  # B L C
        x = self.patch_unembed(x, x_size)

        y = self.norm_Fusion_B(y)  # B L C
        y = self.patch_unembed(y, x_size)
        # x = x.unsqueeze(0)
        # y = y.unsqueeze(0)
        # weights = torch.cat([x, y], 0)
        # weights = self.softmax(weights)
        # wa = weights[0, :]
        # wb = weights[1, :]
        # x = wa * input_x + wb * input_y
        # concatnate x,y in the channel dimension
        x = torch.cat([x, y], 1)
        ## Downsample the feature in the channel dimension
        x = self.lrelu(self.conv_after_body_Fusion(x))
        
        return x

    def forward_features_Re(self, x):        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Re:
            x = layer(x, x_size)

        x = self.norm_Re(x)  # B L C
        x = self.patch_unembed(x, x_size)
        ## Convolution 
        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x) 
        return x

    def forward(self, A, B):
        # print("Initializing the model")
        x = A.unsqueeze(dim=1)
        y = B.unsqueeze(dim=1)
        '''
        shape of x is: torch.Size([8, 1, 84, 84])
        shape of y is: torch.Size([8, 1, 84, 84]) # B, C, H, W
        '''
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        y = self.check_image_size(y)

        self.mean_A = self.mean.type_as(x)
        self.mean_B = self.mean.type_as(y)
        self.mean = (self.mean_A + self.mean_B) / 2

        x = (x - self.mean_A) * self.img_range
        y = (y - self.mean_B) * self.img_range

        # Feedforward
        x = self.forward_features_Ex_A(x)
        y = self.forward_features_Ex_B(y)
        x = self.forward_features_Fusion(x, y)
        x = self.forward_features_Re(x)
        # if self.upsampler == 'pixelshuffle':
        #     # for classical SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.conv_before_upsample(x)
        #     x = self.conv_last(self.upsample(x))
        # elif self.upsampler == 'pixelshuffledirect':
        #     # for lightweight SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.upsample(x)
        # elif self.upsampler == 'nearest+conv':
        #     # for real-world SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.conv_before_upsample(x)
        #     x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #     x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
        #     x = self.conv_last(self.lrelu(self.conv_hr(x)))
        # else:
        #     # for image denoising and JPEG compression artifact reduction
        #     x_1 = self.lrelu(self.conv_first1(x))
        #     x_first = self.lrelu(self.conv_first2(x_1))
        #     res = self.conv_after_body(self.forward_features(x_first)) + x_first
                   
        
        x = x / self.img_range + self.mean
        x = x[:, :, :H*self.upscale, :W*self.upscale] # (batch size, 1, 84, 84)
        x = x.squeeze(dim=1)
        x = self.swin(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers_Ex_A):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Ex_B):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Fusion):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Re):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

    
# if self.multimodality_type == 'cross_attention':    
class Func_Struct_Cross(BaseModel):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=84, patch_size=7, in_chans=1,
                 embed_dim=12, Ex_depths=[6, 6], Fusion_depths=[2, 2, 2], Re_depths=[6,6], 
                 Ex_num_heads=[6, 6], Fusion_num_heads=[6, 6, 6], Re_num_heads=[6, 6],
                 window_size=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(Func_Struct_Cross, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)
        
        ## for fmri processing
        self.concat_method = kwargs.get('concat_method')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.sequence_length = kwargs.get('sequence_length')
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.use_merge_loss = kwargs.get('use_merge_loss')
        self.use_FC = kwargs.get('use_FC')
        # transformer - 일단 seq len이 둘 다 368인 걸 기준으로 짰음. 나중에 또 바꿀 거임.
        # why 128? 368//3과 가장 가까운 16의 배수이기 때문.
        if self.use_merge_loss == True:
            self.transformer_raw = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        
        if self.feature_map_size == 'same':
            self.transformer_low = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
            self.transformer_ultralow = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        elif self.feature_map_size == 'different':
            self.BertConfig_ultralow = BertConfig(hidden_size=self.intermediate_vec, vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=kwargs.get('num_heads_2DBert'), max_position_embeddings=128+1,
                                     hidden_dropout_prob=0.1) 
            self.transformer_ultralow = Transformer_Block(self.BertConfig_ultralow, **kwargs).to(memory_format=torch.channels_last_3d)
            self.transformer_low = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

            if self.feature_map_gen == 'convolution_ul':
                self.proj_u = nn.Conv1d(self.sequence_length, 128, kernel_size=1, padding=0, bias=False)
                
        if self.concat_method == 'concat':
            self.proj_layer = nn.Linear(2*self.intermediate_vec, self.intermediate_vec) 
        self.regression_head = nn.Linear(self.intermediate_vec, self.label_num) #.to(memory_format=torch.channels_last_3d)
        
        
        
        # for DTI+sMRI processing
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        print('in_chans: ', in_chans)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        ## for extracted image ##
        self.swin = SwinTransformerV2(img_size_w=84, img_size_h=84, patch_size=7, in_chans=1, num_classes=1,
                 embed_dim=12, depths=[2, 2, 6], num_heads_swin=[3, 6, 12],
                 window_size=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs
        )
        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        ####修改shallow feature extraction 网络, 修改为2个3x3的卷积####
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim_temp, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.Ex_num_layers = len(Ex_depths)
        self.Fusion_num_layers = len(Fusion_depths)
        self.Re_num_layers = len(Re_depths)

        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed_fusion(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.softmax = nn.Softmax(dim=0)
        # absolute position embedding
        if self.ape: 
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr_Ex = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Ex_depths))]  # stochastic depth decay rule
        dpr_Fusion = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Fusion_depths))]  # stochastic depth decay rule
        dpr_Re = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Re_depths))]  # stochastic depth decay rule
        # build Residual Swin Transformer blocks (RSTB)
        self.layers_Ex_A = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Ex_depths[i_layer],
                         num_heads=Ex_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Ex_A.append(layer)
        self.norm_Ex_A = norm_layer(self.num_features)

        self.layers_Ex_B = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Ex_depths[i_layer],
                         num_heads=Ex_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Ex_B.append(layer)
        self.norm_Ex_B = norm_layer(self.num_features)
        
        self.layers_Fusion = nn.ModuleList()
        for i_layer in range(self.Fusion_num_layers):
            layer = CRSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Fusion_depths[i_layer],
                         num_heads=Fusion_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Fusion[sum(Fusion_depths[:i_layer]):sum(Fusion_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Fusion.append(layer)
        self.norm_Fusion_A = norm_layer(self.num_features)
        self.norm_Fusion_B = norm_layer(self.num_features)
        
        self.layers_Re = nn.ModuleList()
        for i_layer in range(self.Re_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=Re_depths[i_layer],
                         num_heads=Re_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Re[sum(Re_depths[:i_layer]):sum(Re_depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers_Re.append(layer)
        self.norm_Re = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body_Ex_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_Ex_A = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_Fusion = nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_Re = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
            self.conv_last2 = nn.Conv2d(embed_dim_temp, int(embed_dim_temp/2), 3, 1, 1)
            self.conv_last3 = nn.Conv2d(int(embed_dim_temp/2), num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features_Ex_A(self, x):
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))           
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Ex_A:
            x = layer(x, x_size)

        x = self.norm_Ex_A(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features_Ex_B(self, x):    
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))      
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Ex_B:
            x = layer(x, x_size)

        x = self.norm_Ex_B(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features_Fusion(self, x, y):
        input_x = x
        input_y = y        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        
        for layer in self.layers_Fusion:
            x, y = layer(x, y, x_size)
            

        x = self.norm_Fusion_A(x)  # B L C
        x = self.patch_unembed(x, x_size)

        y = self.norm_Fusion_B(y)  # B L C
        y = self.patch_unembed(y, x_size)

        # concatnate x,y in the channel dimension
        x = torch.cat([x, y], 1)
        ## Downsample the feature in the channel dimension
        x = self.lrelu(self.conv_after_body_Fusion(x))
        
        return x

    def forward_features_Re(self, x):        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Re:
            x = layer(x, x_size)

        x = self.norm_Re(x)  # B L C
        x = self.patch_unembed(x, x_size)
        ## Convolution 
        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x) 
        return x

    def compute_fc(self, bold_d):
        bold_d = bold_d.cpu().T.numpy()
        rsFC = np.corrcoef(bold_d)
        rsFC = rsFC * (rsFC>0)
        rsFC = rsFC - np.diag(np.diagonal(rsFC))
        return rsFC
    
    def forward(self, x, x_l, x_u, B):
        # B is struct 
        # 01 fmri process        
        if self.feature_map_size == 'same':
            transformer_dict_low = self.transformer_low(x_l)
            transformer_dict_ultralow = self.transformer_ultralow(x_u)
            
        elif self.feature_map_size == 'different':
            if self.feature_map_gen == 'convolution_ul':
                x_u = self.proj_u(x_u) # torch.Size([1, 128, 84])
            transformer_dict_low = self.transformer_low(x_l)
            transformer_dict_ultralow = self.transformer_ultralow(x_u)
        
        '''
        size of out seq is: torch.Size([1, 361, 84])
        size of out cls is: torch.Size([1, 84])
        size of prediction is: torch.Size([1, 1])
        '''
        
        out_cls_low = transformer_dict_low['cls'] # torch.Size([1, 84])
        out_cls_ultralow = transformer_dict_ultralow['cls'] # torch.Size([1, 84])
        
        if self.concat_method == 'concat':
            out_cls_fmri = torch.cat([out_cls_low, out_cls_ultralow], dim=1) # (1, 84*2)
            out_cls_fmri = self.proj_layer(out_cls_fmri) # (1, 84)
        elif self.concat_method == 'hadamard':
            out_cls_fmri = torch.mul(out_cls_low, out_cls_ultralow) # (1, 84)
        
        device = out_cls_fmri.get_device()
        fmri_embedding = torch.zeros(out_cls_fmri.shape[0], out_cls_fmri.shape[1], out_cls_fmri.shape[1]) # (batch, 84, 84)
        if self.use_FC:
            for i in range(out_cls_fmri.shape[0]):
                rs_FC = self.compute_fc(x[i, :, :]) # (84, 84)
                fmri_embedding_FC = torch.Tensor(rs_FC).to(device) + torch.diag(out_cls_fmri[i, :]) # (84, 84)
                fmri_embedding[i, :, :] = fmri_embedding_FC
        else:
            for i in range(out_cls_fmri.shape[0]):
                fmri_embedding[i, :, :] = torch.diag(out_cls_fmri[i, :]) # (84, 84)
                
        #fmri_embedding = fmri_embedding.unsqueeze(dim=1)
        
        A = fmri_embedding.to(device)
        
        # 02 cross-attention to fmri embedding and DTI+sMRI
        x = A.unsqueeze(dim=1)
        y = B.unsqueeze(dim=1)
        '''
        shape of x is: torch.Size([8, 1, 84, 84])
        shape of y is: torch.Size([8, 1, 84, 84]) # B, C, H, W
        '''
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        y = self.check_image_size(y)

        self.mean_A = self.mean.type_as(x)
        self.mean_B = self.mean.type_as(y)
        self.mean = (self.mean_A + self.mean_B) / 2

        x = (x - self.mean_A) * self.img_range
        y = (y - self.mean_B) * self.img_range

        # Feedforward
        x = self.forward_features_Ex_A(x)
        y = self.forward_features_Ex_B(y)
        x = self.forward_features_Fusion(x, y)
        x = self.forward_features_Re(x)

        
        x = x / self.img_range + self.mean
        x = x[:, :, :H*self.upscale, :W*self.upscale] # (batch size, 1, 84, 84)
        x = x.squeeze(dim=1)
        x = self.swin(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers_Ex_A):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Ex_B):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Fusion):
            flops += layer.flops()
        for i, layer in enumerate(self.layers_Re):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops    
    
    
# Modifying now..
# if self.multimodality_type == 'transfer':
class Func_Struct_Transfer(BaseModel):
    def __init__(self, img_size_w=84, img_size_h=84, patch_size=7, in_chans=1, num_classes=1,
                 embed_dim=12, depths=[2, 2, 6], num_heads_swin=[3, 6, 12],
                 window_size=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super(Func_Struct_Transfer, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)
        self.concat_method = kwargs.get('concat_method')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.sequence_length = kwargs.get('sequence_length')
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.num_classes = num_classes
        self.num_layers = len(depths) # 4
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        ## 01 fMRI
        # why 128? 368//3과 가장 가까운 16의 배수이기 때문.
        if self.feature_map_size == 'same':
            self.transformer_low = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
            self.transformer_ultralow = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        elif self.feature_map_size == 'different':
            self.BertConfig_ultralow = BertConfig(hidden_size=self.intermediate_vec, vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=kwargs.get('num_heads_2DBert'), max_position_embeddings=128+1,
                                     hidden_dropout_prob=0.1) 
            self.transformer_ultralow = Transformer_Block(self.BertConfig_ultralow, **kwargs).to(memory_format=torch.channels_last_3d)
            self.transformer_low = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

            if self.feature_map_gen == 'convolution_ul':
                self.proj_u = nn.Conv1d(self.sequence_length, 128, kernel_size=1, padding=0, bias=False)
                
        if self.concat_method == 'concat':
            self.proj_layer = nn.Linear(2*self.intermediate_vec, self.intermediate_vec) 
        self.regression_head = nn.Linear(self.intermediate_vec, self.label_num) #.to(memory_format=torch.channels_last_3d)
        
        ## 02 DTI+sMRI
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size_w=img_size_w, img_size_h=img_size_h, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches # 441
        patches_resolution = self.patch_embed.patches_resolution  # (21, 21)
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        ## 3번 돌아
        for i_layer in range(self.num_layers):
            input_resolution=(patches_resolution[0] // (2 ** i_layer),
                              patches_resolution[1] // (2 ** i_layer)),
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads_swin=num_heads_swin[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()
        
        
        
    def forward(self, x_l, x_u):
        
        # 01 fmri process

        torch.cuda.nvtx.range_push("transformer")

        
        if self.feature_map_size == 'same':
            transformer_dict_low = self.transformer_low(x_l)
            transformer_dict_ultralow = self.transformer_ultralow(x_u)
            
        elif self.feature_map_size == 'different':
            if self.feature_map_gen == 'convolution_ul':
                x_u = self.proj_u(x_u) # torch.Size([1, 128, 84])
            transformer_dict_low = self.transformer_low(x_l)
            transformer_dict_ultralow = self.transformer_ultralow(x_u)
        
        '''
        size of out seq is: torch.Size([1, 361, 84])
        size of out cls is: torch.Size([1, 84])
        size of prediction is: torch.Size([1, 1])
        '''
        
        out_cls_low = transformer_dict_low['cls'] # torch.Size([1, 84])
        out_cls_ultralow = transformer_dict_ultralow['cls'] # torch.Size([1, 84])
        
        if self.concat_method == 'concat':
            out_cls_fmri = torch.cat([out_cls_low, out_cls_ultralow], dim=1) # (1, 84*2)
            out_cls_fmri = self.proj_layer(out_cls_fmri) # (1, 84)
        elif self.concat_method == 'hadamard':
            out_cls_fmri = torch.mul(out_cls_low, out_cls_ultralow) # (1, 84)
        
        fmri_embedding = torch.zeros(out_cls_fmri.shape[0], out_cls_fmri.shape[1], out_cls_fmri.shape[1]) # (batch, 84, 84)
        for i in range(out_cls_fmri.shape[0]):
            fmri_embedding[i, :, :] = torch.diag(out_cls_fmri[i, :]) # (84, 84)
        
        # 02 training fMRI dataset on DTI+sMRI pre-traiined SwinV2
        
        #struct = struct.unsqueeze(dim=1) # DTI shape: Batch, 1, 84, 84, on the gpu
        fmri_embedding = fmri_embedding.unsqueeze(dim=1)
        device = out_cls_fmri.get_device()

        ## fmri_embedding을 DTI+sMRI에 pretrain된 SwinV2에 통과시키는 것. ##
        fmri_embedding = self.forward_features(fmri_embedding.to(device)) # 얘를 통과하면 (batch, 48)가 됨ㅋㅋㅋ
        prediction = self.head(fmri_embedding) # 얘는 (batch, 1)
        return {self.task:prediction}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

# if self.multimodality_type == 'add':
class Func_Struct_Add(BaseModel):
    def __init__(self, img_size_w=84, img_size_h=84, patch_size=7, in_chans=1, num_classes=1,
                 embed_dim=12, depths=[2, 2, 6], num_heads_swin=[3, 6, 12],
                 window_size=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super(Func_Struct_Add, self).__init__()
        self.task = kwargs.get('fine_tune_task')
        self.register_vars(**kwargs)
        self.concat_method = kwargs.get('concat_method')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.sequence_length = kwargs.get('sequence_length')
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.num_classes = num_classes
        self.num_layers = len(depths) # 4
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        ## 01 fMRI
        # why 128? 368//3과 가장 가까운 16의 배수이기 때문.
        if self.feature_map_size == 'same':
            self.transformer_low = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
            self.transformer_ultralow = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)
        elif self.feature_map_size == 'different':
            self.BertConfig_ultralow = BertConfig(hidden_size=self.intermediate_vec, vocab_size=1,
                                     num_hidden_layers=kwargs.get('transformer_hidden_layers'),
                                     num_attention_heads=kwargs.get('num_heads_2DBert'), max_position_embeddings=128+1,
                                     hidden_dropout_prob=0.1) 
            self.transformer_ultralow = Transformer_Block(self.BertConfig_ultralow, **kwargs).to(memory_format=torch.channels_last_3d)
            self.transformer_low = Transformer_Block(self.BertConfig, **kwargs).to(memory_format=torch.channels_last_3d)

            if self.feature_map_gen == 'convolution_ul':
                self.proj_u = nn.Conv1d(self.sequence_length, 128, kernel_size=1, padding=0, bias=False)
                
        if self.concat_method == 'concat':
            self.proj_layer = nn.Linear(2*self.intermediate_vec, self.intermediate_vec) 
        self.regression_head = nn.Linear(self.intermediate_vec, self.label_num) #.to(memory_format=torch.channels_last_3d)
        
        ## 02 DTI+sMRI
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size_w=img_size_w, img_size_h=img_size_h, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches # 441
        patches_resolution = self.patch_embed.patches_resolution  # (21, 21)
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        ## 3번 돌아
        for i_layer in range(self.num_layers):
            input_resolution=(patches_resolution[0] // (2 ** i_layer),
                              patches_resolution[1] // (2 ** i_layer)),
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads_swin=num_heads_swin[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()
        
        
        
    def forward(self, x_l, x_u, struct):
        
        # 01 fmri process

        torch.cuda.nvtx.range_push("transformer")

        
        if self.feature_map_size == 'same':
            transformer_dict_low = self.transformer_low(x_l)
            transformer_dict_ultralow = self.transformer_ultralow(x_u)
            
        elif self.feature_map_size == 'different':
            if self.feature_map_gen == 'convolution_ul':
                x_u = self.proj_u(x_u) # torch.Size([1, 128, 84])
            transformer_dict_low = self.transformer_low(x_l)
            transformer_dict_ultralow = self.transformer_ultralow(x_u)
        
        '''
        size of out seq is: torch.Size([1, 361, 84])
        size of out cls is: torch.Size([1, 84])
        size of prediction is: torch.Size([1, 1])
        '''
        
        out_cls_low = transformer_dict_low['cls'] # torch.Size([1, 84])
        out_cls_ultralow = transformer_dict_ultralow['cls'] # torch.Size([1, 84])
        
        if self.concat_method == 'concat':
            out_cls_fmri = torch.cat([out_cls_low, out_cls_ultralow], dim=1) # (1, 84*2)
            out_cls_fmri = self.proj_layer(out_cls_fmri) # (1, 84)
        elif self.concat_method == 'hadamard':
            out_cls_fmri = torch.mul(out_cls_low, out_cls_ultralow) # (1, 84)
        
        fmri_embedding = torch.zeros(out_cls_fmri.shape[0], out_cls_fmri.shape[1], out_cls_fmri.shape[1]) # (batch, 84, 84)
        for i in range(out_cls_fmri.shape[0]):
            fmri_embedding[i, :, :] = torch.diag(out_cls_fmri[i, :]) # (84, 84)
        
        # 02 DTI+sMRI process
        
        struct = struct.unsqueeze(dim=1) # DTI shape: Batch, 1, 84, 84, on the gpu
        fmri_embedding = fmri_embedding.unsqueeze(dim=1)
        device = struct.get_device()

        ## struct_embedding이랑 fmri_embedding이랑 더해서 SwinV2 통과시키는 것. 일단은 struct를 쌩으로 넣어놓음. ##
        multimodal = self.forward_features(struct+fmri_embedding.to(device)) # 얘를 통과하면 (batch, 48)가 됨ㅋㅋㅋ
        prediction = self.head(multimodal) # 얘는 (batch, 1)
        
        ans_dict = {self.task:prediction}
        return ans_dict

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

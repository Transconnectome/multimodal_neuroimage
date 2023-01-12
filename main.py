from utils import *  #including 'init_distributed', 'weight_loader'
from trainer import Trainer
import os
from pathlib import Path
import torch

#DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
import builtins

#AMP
from torch.cuda.amp import GradScaler, autocast

#OPTUNA
import optuna 
from copy import deepcopy
import dill
import logging
import sys


def get_arguments(base_path):
    """
    handle arguments from commandline.
    some other hyper parameters can only be changed manually (such as model architecture,dropout,etc)
    notice some arguments are global and take effect for the entire three phase training process, while others are determined per phase
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,default="baseline") 
    parser.add_argument('--dataset_name', type=str, choices=['hcp', 'fMRI_image', 'fMRI_timeseries', 'DTI', 'sMRI', 'struct', 'DTI+sMRI', 'multimodal', 'multimodal_prs'], default="fMRI_timeseries")
    parser.add_argument('--fmri_type', type=str, choices=['timeseries', 'frequency', 'divided_frequency', 'time_domain_low', 'time_domain_ultralow', 'frequency_domain_low', 'frequency_domain_ufltralow', 'timeseries_and_frequency'], default="timeseries")
    parser.add_argument('--intermediate_vec', type=int, choices=[84, 48, 22], default=84)
    parser.add_argument('--fmri_image_path', default='/storage/bigdata/ABCD/fmriprep/1.rs_fmri/6.masked_image') ## lab server
    parser.add_argument('--fmri_timeseries_path', default='/storage/bigdata/ABCD/fmriprep/1.rs_fmri/5.ROI_DATA') ## lab server
    parser.add_argument('--dti_path', default='./data/dti') ## lab server
    parser.add_argument('--hcp_path', default='/scratch/connectome/stellasybae/lowfreqBERT/data/hcp1200/hcp_tc_npy_22')## lab server
    parser.add_argument('--smri_path', default='./data/smri_cortical_thickness') ## lab server
    parser.add_argument('--prs_path', default='./data/prs') ## lab server
    parser.add_argument('--dti+smri_path', default='./data/dti+smri') ## lab server
    parser.add_argument('--base_path', default=base_path)
    parser.add_argument('--step', default='1', choices=['1','2','3','4','5','6'], help='which step you want to run')
    
    parser.add_argument('--voxel_norm_dir', default='per_voxel_normalize', type=str, choices=['per_voxel_normalize','per_voxel_normalize_no_nan', 'global_norm_only'])
    
    
    parser.add_argument('--target', type=str, default='sex', choices=['sex','age','ASD_label','ADHD_label','nihtbx_totalcomp_uncorrected','nihtbx_fluidcomp_uncorrected', 'ADHD_label_robust', 'BMI'],help='fine_tune_task must be specified as follows -- {sex:classification, age:regression, ASD_label:classification, ADHD_label:classification, nihtbx_***:regression}')
    parser.add_argument('--fine_tune_task',
                        default='binary_classification',
                        choices=['regression','binary_classification'],
                        help='fine tune model objective. choose binary_classification in case of a binary classification task')
    parser.add_argument('--seed', type=int, default=55555555)
    
    parser.add_argument('--num_val_samples', type=int, default=1000) #10000이 default. 변화 없음.
    
    
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--log_dir', type=str, default=os.path.join(base_path, 'runs')) #로그는 runs에 저장되는데..?
    parser.add_argument('--random_TR', action='store_false') #True면(인자를 넣어주지 않으면) 전체 sequence 로부터 random sampling(default). False면 (--random_TR 인자를 넣어주면) 0번째 TR부터 sliding window
    
    parser.add_argument('--intensity_factor', default=1)
    
    parser.add_argument('--perceptual_factor', default=1)
    parser.add_argument('--which_perceptual', default='vgg', choices=['vgg','densenet3d'])
    
    parser.add_argument('--reconstruction_factor', default=1)
    parser.add_argument('--transformer_hidden_layers', type=int,default=16)
    #parser.add_argument('--transformer_num_attention_heads',type=int, default=16) # 안 쓰는 듯 함
    #parser.add_argument('--transformer_emb_size',type=int ,default=2640) # 얘도 안 씀
    parser.add_argument('--train_split', default=0.7)
    parser.add_argument('--val_split', default=0.15)
    parser.add_argument('--running_mean_size', default=5000)
    
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--init_method', default='file', type=str, choices=['file','env'], help='DDP init method')
    parser.add_argument('--non_distributed', action='store_true')

    # AMP configs:
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--gradient_clipping', action='store_true')
    #parser.add_argument('--opt_level', default='O1', type=str,
    #                    help='opt level of amp. O1 is recommended')
    
    # Gradient accumulation
    parser.add_argument("--accumulation_steps", default=1, type=int,required=False,help='mini batch size == accumulation_steps * args.train_batch_size')
    
    # Nsight profiling
    parser.add_argument("--profiling", action='store_true')
    
    # multimodality options
    parser.add_argument('--fmri_multimodality_type', default='cross_attention', choices=['cross_attention','two_channels'])
    parser.add_argument('--multimodality_type', choices=['add','cross_attention', 'transfer'])
    
    
    # optuna related 
    parser.add_argument('--use_optuna', action='store_true', help='whether to use optuna hyperparameter training. DB location is determined by exp_name')
    parser.add_argument('--use_best_params_from_optuna', action='store_true', help='load best params from Optuna results in DB. --use_optuna should be False if this argument is True')
    parser.add_argument('--num_trials', type=int, default=10, help='how many trials')
    parser.add_argument('--opt_num_epochs', type=int, default=3, help='how many epochs per trial')
    parser.add_argument('--n_startup_trials', default=2, help='argument for MedianPruner, Pruning is disabled until the given number of trials finish in the same study.')
    parser.add_argument('--n_warmup_steps', default=5, help='argument for MedianPruner, epoch is same as step in our code. Pruning is disabled until the trial exceeds the given number of step. Note that this feature assumes that step starts at zero.')
    parser.add_argument('--interval_steps', default=1, help='argument for MedianPruner, Interval in number of steps between the pruning checks, offset by the warmup steps. If no value has been reported at the time of a pruning check, that particular check will be postponed until a value is reported.')
    
    #wandb related
    parser.add_argument('--wandb_key', default='108101f4b9c3e31a235aa58307d1c6b548cfb54a', type=str,  help='default: key for Stella')
    parser.add_argument('--wandb_mode', default='online', type=str,  help='online|offline')
    
    # optuna related - config for hyperparameter (script 단에서 조절할 수 있도록 함)
    
    parser.add_argument('--hyp_batch_size', action='store_true')
    parser.add_argument('--hyp_lr_gamma', action='store_true')
    
    parser.add_argument('--hyp_lr_init', action='store_true')
    parser.add_argument('--hyp_lr_init_min', default=1e-5)
    parser.add_argument('--hyp_lr_init_ceil', default=1e-3)

    parser.add_argument('--hyp_dropout', action='store_true')
    parser.add_argument('--hyp_dropout_range_small', default=0.1)
    parser.add_argument('--hyp_dropout_range_big', default=0.8)
    
    parser.add_argument('--hyp_vit_dropout', action='store_true')
    
    parser.add_argument('--hyp_vit_attn_dropout', action='store_true')

    parser.add_argument('--hyp_transformer_hidden_layers', action='store_true')
    parser.add_argument('--hyp_transformer_hidden_layers_range_small', default=8)
    parser.add_argument('--hyp_transformer_hidden_layers_range_big', default=16)

    parser.add_argument('--hyp_transformer_num_attention_heads', action='store_true')
    parser.add_argument('--hyp_transformer_num_attention_heads_range_small', default=8)
    parser.add_argument('--hyp_transformer_num_attention_heads_range_big', default=16)

    parser.add_argument('--hyp_weight_decay', action='store_true')
    parser.add_argument('--hyp_weight_decay_min', default=1e-5)
    parser.add_argument('--hyp_weight_decay_ceil', default=1e-2)
    
    # for xgboost
    parser.add_argument('--hyp_min_child_weight', action='store_true')
    parser.add_argument('--hyp_min_child_weight_small', default=1)
    parser.add_argument('--hyp_min_child_weight_big', default=7)    
    
    parser.add_argument('--hyp_max_depth', action='store_true')
    parser.add_argument('--hyp_max_depth_small', default=3)
    parser.add_argument('--hyp_max_depth_big', default=10)
    
    parser.add_argument('--hyp_gamma_xgboost', action='store_true')
    parser.add_argument('--hyp_gamma_xgboost_min', default=0.0)
    parser.add_argument('--hyp_gamma_xgboost_ceil', default=0.4)    
    
    # lowfreqBERT
    
    
    # Model Options
    parser.add_argument('--feature_map_gen', default='convolution_ul+l', choices=['convolution_ul+l','convolution_ul', 'no', 'resample'], help='how to generate feature map: convolution_ul+l|convolution_ul|no')
    parser.add_argument('--feature_map_size', default='same', choices=['same','different'], help='size of feature map: same|different')
    parser.add_argument('--mixing', default='U2L_and_L2U', choices=['U2L_and_L2U', 'U2L', 'L2U'], help='choose model for mixing lowfrq and ultralowfreq: U2L_and_L2U|U2L|L2U')
    parser.add_argument('--concat_method', default='concat', choices=['concat','hadamard'], help='size of feature map: concat|hadamard')
    parser.add_argument('--filtering_type', default='FIR', choices=['FIR', 'Boxcar'])
        
    
    # Tasks
    parser.add_argument('--uonly', action='store_true',
                        help='use the crossmodal fusion into u (default: False)')
    parser.add_argument('--lonly', action='store_true',
                        help='use the crossmodal fusion into l (default: False)')

    # Dropouts
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--attn_dropout_u', type=float, default=0.0,
                        help='attention dropout (for ultralow)')
    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')

    # Architecture
    parser.add_argument('--nlevels', type=int, default=12,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--num_heads_mult', type=int, default=12,
                        help='number of heads for the mutlimodal transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')
    
    
    ##phase 1 2DBERT
    parser.add_argument('--task_phase1', type=str, default='2DBERT')
    parser.add_argument('--batch_size_phase1', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples') #이걸.. 잘게 쪼개볼까? 원래는 4였음.
    parser.add_argument('--validation_frequency_phase1', type=int, default=10000000) # 11 for test #original: 10000) #원래는 1000이었음 -> 약 7분 걸릴 예정.
    parser.add_argument('--nEpochs_phase1', type=int, default=20) #epoch는 10개인 걸로~
    parser.add_argument('--augment_prob_phase1', default=0)
    parser.add_argument('--optim_phase1', default='AdamW')
    parser.add_argument('--weight_decay_phase1', type=float, default=1e-5)
    parser.add_argument('--lr_policy_phase1', default='step', choices=['step','SGDR', 'CosAnn'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase1', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase1', type=float, default=0.97)
    parser.add_argument('--lr_step_phase1', type=int, default=500)
    parser.add_argument('--lr_warmup_phase1', type=int, default=500)
    parser.add_argument('--sequence_length_phase1', type=int ,default=368) # 원래는 1이었음~
    parser.add_argument('--workers_phase1', type=int,default=4)
    parser.add_argument('--num_heads_2DBert', type=int, default=12)
    parser.add_argument('--feature_squeeze', default=False)
    parser.add_argument('--transformer_dropout_rate', type=float, default=0.1) # for phase 1 and 2
    
    
    ##phase 2 lowfreqBERT
    parser.add_argument('--task_phase2', type=str, default='lowfreqBERT')
    parser.add_argument('--batch_size_phase2', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples') #이걸.. 잘게 쪼개볼까? 원래는 4였음.
    parser.add_argument('--validation_frequency_phase2', type=int, default=10000000) # 11 for test #original: 10000) #원래는 1000이었음 -> 약 7분 걸릴 예정.
    parser.add_argument('--nEpochs_phase2', type=int, default=20) #epoch는 10개인 걸로~
    parser.add_argument('--augment_prob_phase2', default=0)
    parser.add_argument('--optim_phase2', default='AdamW')
    parser.add_argument('--weight_decay_phase2', type=float, default=1e-5) # 원래는 1e-7이었음!
    parser.add_argument('--lr_policy_phase2', default='step', choices=['step','SGDR', 'CosAnn'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase2', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase2', type=float, default=0.97)
    parser.add_argument('--lr_step_phase2', type=int, default=500)
    parser.add_argument('--lr_warmup_phase2', type=int, default=500)
    parser.add_argument('--sequence_length_phase2', type=int ,default=368) # 원래는 1이었음~
    parser.add_argument('--workers_phase2', type=int,default=4)
    parser.add_argument('--use_merge_loss', default=False)

    ##phase 3: VIT
    parser.add_argument('--task_phase3', type=str, default='VIT')
    parser.add_argument('--batch_size_phase3', type=int, default=4) #원래는 1이었음
    parser.add_argument('--validation_frequency_phase3', type=int, default=10000000) # 11 for test original: 10000) #원래는 500이었음
    parser.add_argument('--optim_phase3', default='Adam')
    parser.add_argument('--nEpochs_phase3', type=int, default=20)
    parser.add_argument('--augment_prob_phase3', default=0)
    parser.add_argument('--weight_decay_phase3', type=float, default=1e-5)
    parser.add_argument('--lr_policy_phase3', default='step', choices=['step','SGDR', 'CosAnn'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase3', type=float, default=1e-4)
    parser.add_argument('--lr_gamma_phase3', type=float, default=0.97)
    parser.add_argument('--lr_step_phase3', type=int, default=1000)
    parser.add_argument('--lr_warmup_phase3', type=int, default=500)
    #parser.add_argument('--sequence_length_phase3',type=int, default=20)
    parser.add_argument('--workers_phase3', type=int, default=4)
    parser.add_argument('--model_weights_path_phase2', default=None)
    parser.add_argument('--use_cont_loss', default=False)
    parser.add_argument('--use_mask_loss', default=False)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--VIT_name', type=str, default='vit', choices = ['vit', 'swinv2'])
    

    
    ##phase 4 (test)
    parser.add_argument('--task_phase4', type=str, default='test')
    parser.add_argument('--model_weights_path_phase3', default=None)
    parser.add_argument('--batch_size_phase4', type=int, default=4)
    parser.add_argument('--nEpochs_phase4', type=int, default=20)
    parser.add_argument('--augment_prob_phase4', default=0)
    parser.add_argument('--optim_phase4', default='Adam')
    parser.add_argument('--weight_decay_phase4', type=float, default=1e-2)
    parser.add_argument('--lr_policy_phase4', default='step', choices=['step','SGDR', 'CosAnn'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase4', type=float, default=1e-4)
    parser.add_argument('--lr_gamma_phase4', type=float, default=0.9)
    parser.add_argument('--lr_step_phase4', type=int, default=1500)
    parser.add_argument('--lr_warmup_phase4', type=int, default=100)
    parser.add_argument('--sequence_length_phase4', type=int,default=368)
    parser.add_argument('--workers_phase4', type=int, default=4)
    
    
    ##phase 5 Multimodality
    parser.add_argument('--task_phase5', type=str, default='FuncStruct')
    parser.add_argument('--batch_size_phase5', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples') #이걸.. 잘게 쪼개볼까? 원래는 4였음.
    parser.add_argument('--validation_frequency_phase5', type=int, default=10000000) # 11 for test #original: 10000) #원래는 1000이었음 -> 약 7분 걸릴 예정.
    parser.add_argument('--nEpochs_phase5', type=int, default=20) #epoch는 10개인 걸로~
    parser.add_argument('--augment_prob_phase5', default=0)
    parser.add_argument('--optim_phase5', default='AdamW')
    parser.add_argument('--weight_decay_phase5', type=float, default=1e-5)
    parser.add_argument('--lr_policy_phase5', default='step', choices=['step','SGDR', 'CosAnn'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase5', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase5', type=float, default=0.97)
    parser.add_argument('--lr_step_phase5', type=int, default=500)
    parser.add_argument('--lr_warmup_phase5', type=int, default=500)
    parser.add_argument('--sequence_length_phase5', type=int ,default=368) # 원래는 1이었음~
    parser.add_argument('--workers_phase5', type=int,default=4)
    parser.add_argument('--patch_size_phase5', type=int, default=4)
    parser.add_argument('--use_FC', default=False)
    parser.add_argument('--use_unet_loss', action='store_true')
    parser.add_argument('--use_unet_function', action='store_true')
    parser.add_argument('--use_unet_struct', action='store_true')
    parser.add_argument('--use_prs', action='store_true')
    parser.add_argument('--prs_unsqueeze', default='convolution', choices=['single_convolution','multiple_convolution', 'repeat'])


    ##phase 6 MultiVIT
    parser.add_argument('--task_phase6', type=str, default='SwinFusion')
    parser.add_argument('--batch_size_phase6', type=int, default=8, help='for DDP, each GPU processes batch_size_pahse1 samples') #이걸.. 잘게 쪼개볼까? 원래는 4였음.
    parser.add_argument('--validation_frequency_phase6', type=int, default=10000000) # 11 for test #original: 10000) #원래는 1000이었음 -> 약 7분 걸릴 예정.
    parser.add_argument('--nEpochs_phase6', type=int, default=20) #epoch는 10개인 걸로~
    parser.add_argument('--augment_prob_phase6', default=0)
    parser.add_argument('--optim_phase6', default='AdamW')
    parser.add_argument('--weight_decay_phase6', type=float, default=1e-5)
    parser.add_argument('--lr_policy_phase6', default='step', choices=['step','SGDR', 'CosAnn'], help='learning rate policy: step|SGDR')
    parser.add_argument('--lr_init_phase6', type=float, default=1e-3)
    parser.add_argument('--lr_gamma_phase6', type=float, default=0.97)
    parser.add_argument('--lr_step_phase6', type=int, default=500)
    parser.add_argument('--lr_warmup_phase6', type=int, default=500)
    parser.add_argument('--sequence_length_phase6', type=int ,default=368) # 원래는 1이었음~
    parser.add_argument('--workers_phase6', type=int,default=4)
    parser.add_argument('--use_vae', action='store_true')
    parser.add_argument('--use_unet', action='store_true')
    #parser.add_argument('--embed_dim_mult', type=int, default=24)
    
    
    args = parser.parse_args()
    if args.voxel_norm_dir == 'global_norm_only':
        args.voxel_norm_dir = None
        
    return args

def setup_folders(base_path): 
    os.makedirs(os.path.join(base_path,'experiments'),exist_ok=True) 
    os.makedirs(os.path.join(base_path,'runs'),exist_ok=True)
    os.makedirs(os.path.join(base_path, 'splits'), exist_ok=True)
    return None

def run_phase(args,loaded_model_weights_path,phase_num,phase_name):
    """
    main process that runs each training phase
    :return path to model weights (pytorch file .pth) aquried by the current training phase
    """
    experiment_folder = '{}_{}_{}_{}'.format(args.dataset_name,phase_name,args.target,args.exp_name)
    experiment_folder = Path(os.path.join(args.base_path,'experiments',experiment_folder))
    os.makedirs(experiment_folder, exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num,loaded_model_weights_path)
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    
    print(f'saving the results at {args.experiment_folder}')
    
    # save hyperparameters
    args_logger(args)
    
    # make args to dict. + detach phase numbers from args
    kwargs = sort_args(phase_num, vars(args))

    S = ['train','val']

    if kwargs.get('use_optuna') == True:
        # referred to these links
        # https://python-bloggers.com/2022/08/hyperparameter-tuning-a-transformer-with-optuna/
        if kwargs.get('hyp_lr_init'):
            LR_MIN = kwargs.get('hyp_lr_init_min') #1e-6
            LR_CEIL = kwargs.get('hyp_lr_init_ceil') #1e-3
        if kwargs.get('hyp_weight_decay'):
            WD_MIN = kwargs.get('hyp_weight_decay_min') #1e-5
            WD_CEIL = kwargs.get('hyp_weight_decay_ceil') #1e-2
        
        if kwargs.get('hyp_transformer_hidden_layers'):
            TF_HL_small = kwargs.get('hyp_transformer_hidden_layers_range_small') #8
            TF_HL_big = kwargs.get('hyp_transformer_hidden_layers_range_big') #16
            TF_HL = [TF_HL_small, TF_HL_big]
        
        if kwargs.get('hyp_transformer_num_attention_heads'):
            TF_AH_small = kwargs.get('hyp_transformer_num_attention_heads_range_small') #8
            TF_AH_big = kwargs.get('hyp_transformer_num_attention_heads_range_big') #16
            TF_AH = [TF_AH_small, TF_AH_big]


        if kwargs.get('hyp_dropout'):    
            DO_small = kwargs.get('hyp_dropout_range_small') #0.1
            DO_big = kwargs.get('hyp_dropout_range_big') #0.8
            DO = [DO_small, DO_big]
        #Validation_Frequency = 69
        # 69 for batch size 16 and world size 40
        # same as iteration
        NUM_EPOCHS = kwargs.get('opt_num_epochs') # each trial undergo 'opt_num_epochs' epochs
        is_classification = kwargs.get('fine_tune_task') == 'binary_classification'

        def objective(single_trial: optuna.Trial): 
            # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
            device = torch.device('cuda') #int(os.environ["LOCAL_RANK"]))
            trial = single_trial #optuna.integration.pytorch_distributed.TorchDistributedTrial(single_trial, device=device)

            # The code below should be changed for hyperparameter tuning
            trial_kwargs = deepcopy(kwargs)
            trial_kwargs['lr_step'] = 500
            if kwargs.get('hyp_batch_size'):
                trial_kwargs['batch_size'] = trial.suggest_int("batch_size",low=4, high=16, step=4)
            if kwargs.get('hyp_lr_init'):
                trial_kwargs['lr_init'] = trial.suggest_float("lr_init",low=LR_MIN, high=LR_CEIL, log=True)
            if kwargs.get('hyp_lr_gamma'):
                trial_kwargs['lr_gamma'] = trial.suggest_float("lr_gamma",low=0.1, high=0.9)
            if kwargs.get('hyp_weight_decay'):
                trial_kwargs['weight_decay'] = trial.suggest_float('weight_decay', low=WD_MIN, high=WD_CEIL, log=True)

            # model related
            if kwargs.get('hyp_transformer_hidden_layers'):
                trial_kwargs['transformer_hidden_layers'] = trial.suggest_categorical('transformer_hidden_layers', choices=TF_HL)
            if kwargs.get('hyp_transformer_num_attention_heads'):
                trial_kwargs['transformer_num_attention_heads'] = trial.suggest_categorical('transformer_num_attention_heads', choices=TF_AH)
            if kwargs.get('hyp_seq_len'):
                trial_kwargs['sequence_length'] = trial.suggest_categorical('sequence_length', choices=SL)
            if kwargs.get('hyp_dropout'):
                trial_kwargs['transformer_dropout_rate'] = trial.suggest_float('transformer_dropout_rate', low = 0.1, high=0.8, step=0.1)
            if kwargs.get('hyp_vit_dropout'):
                trial_kwargs['drop_rate'] = trial.suggest_float('drop_rate', low = 0.0, high=0.8, step=0.1)
            if kwargs.get('hyp_vit_attn_dropout'):
                trial_kwargs['attn_drop_rate'] = trial.suggest_float('attn_drop_rate', low = 0.0, high=0.8, step=0.1)    
            trial_kwargs['nEpochs'] = NUM_EPOCHS
            trial_kwargs['trial'] = trial
            trainer = Trainer(sets=S,**trial_kwargs)

            # classification
            best_val_AUROC, best_val_loss = trainer.training()

            return best_val_AUROC if is_classification else best_val_loss


        #----------------------------------------------------------------------------------------------------
        #                    CREATE OPTUNA STUDY
        #----------------------------------------------------------------------------------------------------

        study_name = args.exp_name
        print('NUM_TRIALS:', args.num_trials)
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        if args.rank == 0:
            print('Triggering Optuna study')
            print('study_name:',study_name)
            # storage=optuna.storages.RDBStorage(
            # url='postgresql://junbeom_admin:DBcase6974!@nerscdb03.nersc.gov/junbeom', #"sqlite:///{}.db".format(study_name),
            # skip_compatibility_check=True
            # )
            storage=optuna.storages.RDBStorage(
            url="sqlite:///{}.db".format(study_name),
            engine_kwargs={ "connect_args": {"timeout": 10}},
            skip_compatibility_check=True
            )
            # Default is TPESampler 
            study = optuna.create_study(study_name=study_name, pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps) ,storage=storage, load_if_exists=True, direction='maximize' if is_classification else 'minimize') 
            study.optimize(objective, n_trials=args.num_trials)  
        else:
            for _ in range(args.num_trials):
                try:
                    objective(None)
                except optuna.TrialPruned:
                    pass
        
        # with DDP, each process (ranks) undergo 'NUM_TRIALS' trails
        # so, total NUM_TRIALS * world_size would be run (20 * 40 = 800)

        if args.rank == 0:
            assert study is not None
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            print("Study statistics: ")
            print("Number of finished trials: ", len(study.trials))
            print("Number of pruned trials: ", len(pruned_trials))
            print("Number of complete trials: ", len(complete_trials))

            print('Finding study best parameters')
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
                print('replace hyperparameter with best hyperparameters')
                if key == 'learning_rate':
                    kwargs['lr_init'] = value
                elif key == 'gamma':
                    kwargs['lr_gamma'] = value
                else:
                    kwargs[key] = value

            #kwargs to pkl
            with open(os.path.join(args.experiment_folder,'best_arguments.pkl'),'wb') as f:
                dill.dump(kwargs,f)
            
            #kwargs to txt
            with open(os.path.join(args.experiment_folder,'best_argument_documentation.txt'),'w+') as f:
                for name,arg in kwargs.items():
                    f.write('{}: {}\n'.format(name,arg))

    else:
        if kwargs.get('use_best_params_from_optuna') == True:  # args.use_optuna should be False 
            print('use_best_params_from_optuna')
            study_name = args.exp_name
            storage=optuna.storages.RDBStorage(
                    url="sqlite:///{}.db".format(study_name),
                    engine_kwargs={ "connect_args": {"timeout": 10}},
                    skip_compatibility_check=True
                    )
            is_classification = kwargs.get('fine_tune_task') == 'binary_classification'
            study = optuna.create_study(study_name=study_name, pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps, interval_steps=args.interval_steps) ,storage=storage, load_if_exists=True, direction='maximize' if is_classification else 'minimize')
            for key,value in study.best_params.items():
                if key == 'learning_rate':
                    print(f"replacing the value of learning_rate : from {kwargs['lr_init']} to {value}")
                    kwargs['lr_init'] = value
                elif key == 'gamma':
                    print(f"replacing the value of gamma : from {kwargs['lr_gamma']} to {value}")
                    kwargs['lr_gamma'] = value
                else:
                    print(f'replacing the value of {key} : from {kwargs[key]} to {value}')
                    kwargs[key] = value
                
            kwargs['lr_step'] = 500 # fix the hyperparameter for fixed periods.

        trainer = Trainer(sets=S,**kwargs)
        trainer.training()
                          
        S = ['train','val']

        if phase_num == '3' and not fine_tune_task == 'regression':
            critical_metric = 'accuracy'
        else:
            critical_metric = 'loss'
        model_weights_path = os.path.join(trainer.writer.experiment_folder,trainer.writer.experiment_title + '_BEST_val_{}.pth'.format(critical_metric)) 
                          
        return model_weights_path


'''
def run_phase(args,loaded_model_weights_path,phase_num,phase_name):
    """
    main process that runs each training phase
    :return path to model weights (pytorch file .pth) aquried by the current training phase
    """
    experiment_folder = '{}_{}_{}_{}'.format(args.dataset_name,phase_name,args.target,args.exp_name)
    experiment_folder = Path(os.path.join(args.base_path,'experiments',experiment_folder))
    os.makedirs(experiment_folder, exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num,loaded_model_weights_path)
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    
    
    fine_tune_task = args.fine_tune_task
    print(f'saving the results at {args.experiment_folder}')
    args_logger(args)
    args = sort_args(phase_num, vars(args))
    S = ['train','val']
    trainer = Trainer(sets=S,**args)
    trainer.training()
    if phase_num == '3' and not fine_tune_task == 'regression':
        critical_metric = 'accuracy'
    else:
        critical_metric = 'loss'
    model_weights_path = os.path.join(trainer.writer.experiment_folder,trainer.writer.experiment_title + '_BEST_val_{}.pth'.format(critical_metric))
    
    return model_weights_path

'''

def test(args,phase_num,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), args.exp_name) #, datestamp())
    experiment_folder = Path(os.path.join(args.base_path,'tests', experiment_folder))
    os.makedirs(experiment_folder,exist_ok=True)
    setattr(args,'loaded_model_weights_path_phase' + phase_num, model_weights_path) # 이름이 이게 맞나?
    
    args.experiment_folder = experiment_folder
    args.experiment_title = experiment_folder.name
    args_logger(args)
    args = sort_args(args.step, vars(args))
    S = ['test']
    #trainer = Trainer(experiment_folder, '3', args, ['test'], model_weights_path)
    trainer = Trainer(sets=S,**args) # trainer? 왜 test가 안 들어가지??
    trainer.testing()
    '''
    print(args.fine_tune_task)
    if not args.fine_tune_task == 'regression':
        critical_metric = 'accuracy'
    else:
        critical_metric = 'loss'
    model_weights_path = os.path.join(trainer.writer.experiment_folder,trainer.writer.experiment_title + '_BEST_test_{}.pth'.format(critical_metric))
    '''

''' 기존 함수
def test(args,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), datestamp())
    experiment_folder = os.path.join(args.base_path,'tests', experiment_folder)
    os.makedirs(experiment_folder,exist_ok=True)
    trainer = Trainer(experiment_folder, '3', args, ['test'], model_weights_path)
    trainer.testing()
'''

if __name__ == '__main__':
    base_path = os.getcwd() 
    setup_folders(base_path) 
    args = get_arguments(base_path)

    # DDP initialization
    init_distributed(args)

    # load weights that you specified at the Argument
    model_weights_path, step, task = weight_loader(args)

    if step == '4' :
        print(f'starting testing')
        phase_num = '4'
        test(args, phase_num, model_weights_path) # have some problems here - I checked it! -Stella 
    else:
        print(f'starting phase{step}: {task}')
        run_phase(args,model_weights_path,step,task)
        print(f'finishing phase{step}: {task}')

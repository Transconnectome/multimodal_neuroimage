import numpy as np
import pandas as pd
import scipy.io
import random
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
import nibabel as nib
import scipy

import torch.nn.functional as F
import torchaudio.functional as AF

import nitime

# Import the time-series objects:
from nitime.timeseries import TimeSeries

# Import the analysis objects:
from nitime.analysis import SpectralAnalyzer, FilterAnalyzer, NormalizationAnalyzer

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
    def register_args(self,**kwargs):
        #todo:decide if keep immedieate load or not
        self.device = None #torch.device('cuda:{}'.format(kwargs.get('gpu')) if kwargs.get('cuda') else torch.device('cpu')) 
        self.index_l = []
        self.target = kwargs.get('target')
        self.fine_tune_task = kwargs.get('fine_tune_task')
        self.dataset_name = kwargs.get('dataset_name')
        self.fmri_type = kwargs.get('fmri_type')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.set_augmentations(**kwargs)
#         self.stride_factor = 1
#         self.sequence_stride = 1 # 어느 정도 주기로 volume을 샘플링 할 것인가
#         self.sequence_length = kwargs.get('sequence_length') # 몇 개의 volume을 사용할 것인가(STEP1:1,STEP2:20,STEP3:20 마다 다름)
#         self.sample_duration = self.sequence_length * self.sequence_stride #샘플링하는 대상이 되는 구간의 길이 
#         self.stride = max(round(self.stride_factor * self.sample_duration),1) # sequence lenghth 만큼씩 이동해서 rest_1200_3D의 init 파트의 for문에서 TR index를 불러옴
#         self.TR_skips = range(0,self.sample_duration,self.sequence_stride)

    def set_augmentations(self,**kwargs):
        if kwargs.get('augment_prob') > 0:
            self.augment = augmentations.brain_gaussian(**kwargs)
        else:
            self.augment = None
    
    def get_input_shape(self):
        if self.dataset_name == 'fMRI_image':
            shape = nib.load(self.index_l[0][2]).get_fdata().shape # shape is: (99, 117, 95, 363)
        elif self.dataset_name in ['fMRI_timeseries', 'DTI', 'sMRI', 'struct', 'hcp', 'DTI+sMRI', 'multimodal']:
            shape = np.load(self.index_l[0][2]).shape # shape is: (370, 84) in fMRI timeseries case
        return shape



class HCP_fMRI_timeseries(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('hcp_path')
        
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.target = kwargs.get('target')
        
        # HCP 에서 target value가 결측값인 샘플 제거
        
        if self.target == 'sex':
            self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','HCP_1200_gender.csv'))
            non_na = self.meta_data[['Subject', 'Gender']].dropna(axis=0)
            subjects = list(non_na['Subject']) # 100206 형식. metadata 기준~
            subjects = list(map(str, subjects)) 
        elif self.target == 'age':
            self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','HCP_1200_precise_age.csv'))
            non_na = self.meta_data[['subject', 'age']].dropna(axis=0)
            subjects = list(non_na['subject']) # 100206 형식. metadata 기준~
            subjects = list(map(int, subjects))
            subjects = list(map(str, subjects)) 
        else:
            non_na = self.meta_data[['Subject', self.target]].dropna(axis=0)
        
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        for i,subject in enumerate(os.listdir(self.data_dir)):
            # subject의 형식: 100206_cortex.npy
            subject = subject.split('_')[0]
            # subject의 형식: 100206
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    if self.target == 'age':
                        target = torch.tensor((self.meta_data.loc[self.meta_data['subject']==int(subject), 'age'].values[0] - cont_mean) / cont_std)
                        target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    if self.target == 'sex':
                        target = self.meta_data.loc[self.meta_data['Subject']==int(subject), 'Gender'].values[0]
                        target = 1.0 if target == 'M' else 0
                        target = torch.tensor(target)

                path_to_fMRIs = os.path.join(self.data_dir, subject+'_cortex.npy') # npy 파일로 접근할 것
                        
                #if np.load(path_to_fMRIs)[20:].T.shape[1]>= 350:
                self.index_l.append((i, subject, path_to_fMRIs, target))
                
    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]
        y = np.load(path_to_fMRIs) # (22, 1200)
        ts_length = y.shape[1]
        pad = 1200 - ts_length
        if self.augment is not None:
            y = self.augment(y)
        y = scipy.stats.zscore(y, axis=None) # (22, 922 ~ 1200)
        y = F.pad(torch.from_numpy(y), (pad//2, pad-pad//2), "constant", 0) # (22, 1200)
        y = y.T.float() #.type(torch.DoubleTensor) # (1200, 22)
        ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}
        
        return ans_dict

    
    
class ABCD_fMRI_image(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('fmri_image_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        
        # ABCD 에서 target value가 결측값인 샘플 제거
        non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
        
        subjects = list(non_na['subjectkey']) # 여기는 잘 됨!!
        
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        for i,subject in enumerate(os.listdir(self.data_dir)):
            # subject의 형식: masked_image_sub-NDARINVTF8HPRGG.nii.gz
            subject = subject.split('-')[1].split('.')[0]
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                    target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
                
                path_to_fMRIs = os.path.join(self.data_dir,'masked_image_sub-'+subject+'.nii.gz') # nii.gz. 파일로 접근할 것
                self.index_l.append((i, subject, path_to_fMRIs, target))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]
        y = nib.load(path_to_fMRIs).get_fdata()
        if self.augment is not None:
            y = self.augment(y)
        return {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}

class ABCD_fMRI_timeseries(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.data_dir = kwargs.get('fmri_timeseries_path')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.filtering_type = kwargs.get('filtering_type')
        
        # ABCD 에서 target value가 결측값인 샘플 제거
        non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
        
        #subjects = list(non_na['subjectkey']) # subjects의 형식: NDARINVZRHTXMXD #나중에 그냥 이걸 고치자..
        with open("rsfMRI_upper370_sub_list.txt", mode="r") as file:
            subject_upper370 = file.read().splitlines()
        
        subjects = list(set(list(non_na['subjectkey'])) & set(subject_upper370))
        
#         if 'resample' in self.feature_map_gen:
#             with open("resample_nan_sub_list.txt", mode="r") as file:
#                 nan_subjects = file.read().splitlines()
#             subjects = set(subjects) - set(nan_subjects)
            
        with open("rsfMRI_filtering_with_nan_sub_list.txt", mode="r") as file:
            nan_subjects = file.read().splitlines()
        subjects = set(subjects) - set(nan_subjects)
        
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        for i,subject in enumerate(os.listdir(self.data_dir)):
            # subject의 형식: sub-NDARINVZRHTXMXD
            subject=subject.split('-')[1]
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                    target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
                
                if self.dataset_name == 'fMRI_timeseries':
                    if self.intermediate_vec == 84:
                        path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'desikankilliany_sub-'+subject+'.npy') # npy 파일로 접근할 것 
                    elif self.intermediate_vec == 48:
                        path_to_fMRIs = os.path.join(self.data_dir, 'sub-'+subject, 'harvard_oxford_sub-'+subject+'.npy') # npy 파일로 접근할 것
                        
                #if np.load(path_to_fMRIs)[20:].T.shape[1]>= 350:
                self.index_l.append((i, subject, path_to_fMRIs, target))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_fMRIs, target = self.index_l[index]
        if self.dataset_name == 'fMRI_timeseries':
            y = np.load(path_to_fMRIs)[20:].T # [84, 350 ~ 361]
            ts_length = y.shape[1]
            pad = 368-ts_length
            if self.augment is not None:
                y = self.augment(y)

            if self.fmri_type == 'timeseries':
                y = scipy.stats.zscore(y, axis=None) # (84, 350 ~ 361)
                y = F.pad(torch.from_numpy(y), (pad//2, pad-pad//2), "constant", 0) # (84, 361)
                y = y.T.float() #.type(torch.DoubleTensor) # (361, 84)
                ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}
            
            elif self.fmri_type == 'frequency':
                T = TimeSeries(y, sampling_interval=0.8)
                S_original = SpectralAnalyzer(T)
                y = scipy.stats.zscore(np.abs(S_original.spectrum_fourier[1]), axis=None) #(84, 177)
                #if np.isnan(y).sum() == 0:
                pad = 184 - y.shape[1] # 8의 배수로 맞춰주기 위함.. 인데 굳이 해야 하나?
                y = F.pad(torch.from_numpy(y), (pad//2, pad-pad//2), "constant", 0) # (84, 184)
                y = y.T.float() # (184, 84)
                ans_dict = {'fmri_sequence':y,'subject':subj,'subject_name':subj_name, self.target:target}
                
            elif self.fmri_type == 'time_domain_low':
                T = TimeSeries(y, sampling_interval=0.8)
                FA = FilterAnalyzer(T, lb=0.0035)
                low = scipy.stats.zscore(FA.fir.data, axis=1) #1) #(84, 353)
                #if np.isnan(low).sum() == 0:
                # pad - low                                
                low = F.pad(torch.from_numpy(low), (pad//2, pad-pad//2), "constant", 0) # (84, 368)
                low = low.T.float() #.type(torch.DoubleTensor)  # (368, 84)
                ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}
            
            elif self.fmri_type == 'time_domain_ultralow':
                T = TimeSeries(y, sampling_interval=0.8)
                FA = FilterAnalyzer(T, lb=0.0035)
                ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1) #1) #(84, 353)
                #if np.isnan(ultralow).sum() == 0:
                # pad - ultralow
                if self.feature_map_gen == 'resample':
                    ultralow = AF.resample(waveform = torch.Tensor(ultralow),
                                           orig_freq = 3,
                                           new_freq = 1,
                                           resampling_method = 'sinc_interpolation')
                    pad_ultralow = 128 - ultralow.shape[1]
                    ultralow = F.pad(ultralow, (pad_ultralow//2, pad_ultralow-pad_ultralow//2), "constant", 0) # (84, 128)
                else:
                    ultralow = F.pad(torch.from_numpy(ultralow), (pad//2, pad-pad//2), "constant", 0) # (84, 368)
                
                ultralow = ultralow.T.float() #.type(torch.DoubleTensor)  # (368, 84)
                ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
                
            elif self.fmri_type == 'divided_frequency':

                # frequency divide - padding ㄱㄱ
                T = TimeSeries(y, sampling_interval=0.8)
                FA = FilterAnalyzer(T, lb=0.0035)
                raw = scipy.stats.zscore(FA.data, axis=1)
                if self.filtering_type == 'FIR':
                    low = scipy.stats.zscore(FA.fir.data, axis=1) #1) #(84, 353)
                    ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1) #1) #(84, 353)
                elif self.filtering_type == 'Boxcar':
                    low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1) #1) #(84, 353)
                    ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1) #1) #(84, 353)
                
                # pad - raw
                raw = F.pad(torch.from_numpy(raw), (pad//2, pad-pad//2), "constant", 0) # (84, 368)
                raw = raw.T.float()
                
                #if np.isnan(low).sum() == 0:
                # pad - low                                
                low = F.pad(torch.from_numpy(low), (pad//2, pad-pad//2), "constant", 0) # (84, 368)
                low = low.T.float() #.type(torch.DoubleTensor)  # (368, 84)

                # pad - ultralow
                if self.feature_map_gen == 'resample' and self.feature_map_size == 'different':
                    ultralow = AF.resample(waveform = torch.Tensor(ultralow),
                                           orig_freq = 3,
                                           new_freq = 1,
                                           resampling_method = 'sinc_interpolation') # torch.Size([84, 117])
                    pad_ultralow = 128 - ultralow.shape[1]
                    ultralow = F.pad(ultralow, (pad_ultralow//2, pad_ultralow-pad_ultralow//2), "constant", 0) # (84, 128) - pad on last dimension
                else:
                    ultralow = F.pad(torch.from_numpy(ultralow), (pad//2, pad-pad//2), "constant", 0) # (84, 368)
                
                ultralow = ultralow.T.float() #.type(torch.DoubleTensor)  # (368, 84) or (128, 84)

                ans_dict= {'fmri_sequence':raw, 'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'subject':subj, 'subject_name':subj_name, self.target:target}
            elif self.fmri_type == 'frequency_domain_low':
                # frequency divide - padding ㄱㄱ
                T = TimeSeries(y, sampling_interval=0.8)
                FA = FilterAnalyzer(T, lb=0.0035)
                
                # pad - low (frequency)
                T1 = TimeSeries((FA.fir.data), sampling_interval=0.8)
                S_original1 = SpectralAnalyzer(T1)
                #ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:]) # (176, 84)
                #ultralow = torch.from_numpy(ultralow).float() # (176, 84)

                low = np.abs(S_original1.spectrum_fourier[1].T[1:].T) # (84, 176)
                pad_l = 184 - low.shape[1]
                low = F.pad(torch.from_numpy(low), (pad_l//2, pad_l-pad_l//2), "constant", 0) # (84, 184)
                low = low.T.float() #.type(torch.DoubleTensor)  # (184, 84)
                ans_dict = {'fmri_sequence':low,'subject':subj,'subject_name':subj_name, self.target:target}
           
            elif self.fmri_type == 'frequency_domain_ultralow':
                # frequency divide - padding ㄱㄱ
                T = TimeSeries(y, sampling_interval=0.8)
                FA = FilterAnalyzer(T, lb=0.0035)

                # pad - ultralow (frequency)
                T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=0.8)
                S_original1 = SpectralAnalyzer(T1)
                #ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:]) # (176, 84)
                #ultralow = torch.from_numpy(ultralow).float() # (176, 84)

                ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T) # (84, 176)
                pad_u = 184 - ultralow.shape[1]
                ultralow = F.pad(torch.from_numpy(ultralow), (pad_u//2, pad_u-pad_u//2), "constant", 0) # (84, 184)
                ultralow = ultralow.T.float() #.type(torch.DoubleTensor)  # (184, 84)    
            
                ans_dict = {'fmri_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
            elif self.fmri_type == 'timeseries_and_frequency':
                # frequency divide - padding ㄱㄱ
                T = TimeSeries(y, sampling_interval=0.8)
                FA = FilterAnalyzer(T, lb=0.0035)
                low = scipy.stats.zscore(FA.fir.data, axis=1) #1) #(84, 353)
                ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1) #1) #(84, 353)
                
                #if np.isnan(low).sum() == 0:
                # pad - low (timeseries)                               
                low = F.pad(torch.from_numpy(low), (pad//2, pad-pad//2), "constant", 0) # (84, 368)
                low = low.T.float() #.type(torch.DoubleTensor)  # (368, 84)

                # pad - ultralow (frequency)
                T1 = TimeSeries((FA.data-FA.fir.data), sampling_interval=0.8)
                S_original1 = SpectralAnalyzer(T1)
                #ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:]) # (176, 84)
                #ultralow = torch.from_numpy(ultralow).float() # (176, 84)

                ultralow = np.abs(S_original1.spectrum_fourier[1].T[1:].T) # (84, 176)
                pad_u = 184 - ultralow.shape[1]
                ultralow = F.pad(torch.from_numpy(ultralow), (pad_u//2, pad_u-pad_u//2), "constant", 0) # (84, 184)
                ultralow = ultralow.T.float() #.type(torch.DoubleTensor)  # (184, 84)

                ans_dict= {'fmri_lowfreq_sequence':low,'fmri_ultralowfreq_sequence':ultralow,'subject':subj,'subject_name':subj_name, self.target:target}
        
        return ans_dict

class ABCD_DTI(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('dti_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        #self.intermediate_vec = kwargs.get('intermediate_vec')
        
        # ABCD 에서 target value가 결측값인 샘플 제거
        non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
        subjects = list(non_na['subjectkey']) # subjects의 형식: NDARINVZRHTXMXD

        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        
        for i,filename in enumerate(os.listdir(self.data_dir)):
            subject=filename.split('_')[-1].split('.')[0] # subject의 형식: NDARINVZRHTXMXD
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                    target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
                
                path_to_DTIs = os.path.join(self.data_dir, filename)
                self.index_l.append((i, subject, path_to_DTIs, target))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_DTIs, target = self.index_l[index]
        y = np.load(path_to_DTIs)
        #y = torch.Tensor(y).half()
        y = torch.Tensor(scipy.stats.zscore(y, axis=None)).half()
        ans_dict= {'dti':y, 'subject':subj,'subject_name':subj_name, self.target:target}
        
        return ans_dict

class ABCD_sMRI(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('smri_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        #self.intermediate_vec = kwargs.get('intermediate_vec')
        
        # ABCD 에서 target value가 결측값인 샘플 제거
        non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
        subjects = list(non_na['subjectkey']) # subjects의 형식: NDARINVZRHTXMXD

        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        
        for i,filename in enumerate(os.listdir(self.data_dir)):
            subject=filename.split('_')[-1].split('.')[0] # subject의 형식: NDARINVZRHTXMXD
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                    target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
                
                path_to_sMRIs = os.path.join(self.data_dir, filename)
                self.index_l.append((i, subject, path_to_sMRIs, target))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_sMRIs, target = self.index_l[index]
        y = np.load(path_to_sMRIs)
        #y = torch.Tensor(y).half()
        y = torch.Tensor(scipy.stats.zscore(y, axis=None)).half()
        ans_dict= {'smri':y, 'subject':subj,'subject_name':subj_name, self.target:target}
        
        return ans_dict
    
class ABCD_struct (BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.smri_dir = kwargs.get('smri_path')
        self.dti_dir = kwargs.get('dti_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        
        self.subject_folders = []
        #self.intermediate_vec = kwargs.get('intermediate_vec')
        
        # DTI와 sMRI가 같은 subject
        with open("DTI_sMRI_intersection_sub_list.txt", mode="r") as file:
            DTI_sMRI_inter = file.read().splitlines()
        
        # ABCD 에서 target value가 결측값인 샘플 제거
        non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
        subjects = list(set(list(non_na['subjectkey'])) & set(DTI_sMRI_inter)) # subjects의 형식: NDARINVZRHTXMXD
        
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        
        for i,smriname in enumerate(os.listdir(self.smri_dir)):
            subject=smriname.split('_')[-1].split('.')[0] # subject의 형식: NDARINVZRHTXMXD, filename의 형식 : 'smri_NDARINVKPNEX4HY.npy'
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                    target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
                
                path_to_sMRIs = os.path.join(self.smri_dir, smriname)
                dtiname = 'dti_count_'+subject+'.npy'
                path_to_DTIs = os.path.join(self.dti_dir, dtiname)
                self.index_l.append((i, subject, path_to_sMRIs, path_to_DTIs, target))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_sMRIs,  path_to_DTIs, target = self.index_l[index]
        dti = np.load(path_to_DTIs)
        sMRI = np.load(path_to_sMRIs)
        dti = torch.Tensor(scipy.stats.zscore(dti, axis=None)).half()
        sMRI = torch.Tensor(scipy.stats.zscore(sMRI, axis=None)).half()
        ans_dict= {'smri':sMRI, 'dti': dti, 'subject':subj,'subject_name':subj_name, self.target:target}
        
        return ans_dict
    
class ABCD_DTI_sMRI(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.data_dir = kwargs.get('dti+smri_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        #self.intermediate_vec = kwargs.get('intermediate_vec')
        
        # ABCD 에서 target value가 결측값인 샘플 제거
        non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
        subjects = list(non_na['subjectkey']) # subjects의 형식: NDARINVZRHTXMXD

        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        
        for i,filename in enumerate(os.listdir(self.data_dir)):
            # filename 형식: dti_count+smri_cortical_thickness_NDARINVZRHTXMXD.npy
            subject=filename.split('_')[-1].split('.')[0] # subject의 형식: NDARINVZRHTXMXD
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                    target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
                
                path_to_DTI_sMRIs = os.path.join(self.data_dir, filename)
                self.index_l.append((i, subject, path_to_DTI_sMRIs, target))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_DTI_sMRIs, target = self.index_l[index]
        y = np.load(path_to_DTI_sMRIs)
        y = torch.Tensor(scipy.stats.zscore(y, axis=None)).half()
        #y = torch.Tensor(y).half()
        ans_dict= {'struct':y, 'subject':subj,'subject_name':subj_name, self.target:target}
        
        return ans_dict
    
    
class ABCD_multimodal(BaseDataset):
    def __init__(self, **kwargs):
        self.register_args(**kwargs)
        self.struct_dir = kwargs.get('dti+smri_path')
        self.fmri_dir = kwargs.get('fmri_timeseries_path')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_names = os.listdir(self.struct_dir)
        self.subject_folders = []
        self.feature_map_gen = kwargs.get('feature_map_gen')
        self.feature_map_size = kwargs.get('feature_map_size')
        self.meta_data = pd.read_csv(os.path.join(kwargs.get('base_path'),'data','metadata','ABCD_phenotype_total.csv'))
        self.subject_folders = []
        self.intermediate_vec = kwargs.get('intermediate_vec')
        self.filtering_type = kwargs.get('filtering_type')
        #self.intermediate_vec = kwargs.get('intermediate_vec')
        
        # ABCD 에서 target value가 결측값인 샘플 제거
        non_na = self.meta_data[['subjectkey',self.target]].dropna(axis=0)
        subjects = list(non_na['subjectkey']) # subjects의 형식: NDARINVZRHTXMXD
        
        with open("rsfMRI_upper370_sub_list.txt", mode="r") as file:
            subject_upper370 = file.read().splitlines()
        
        subjects = list(set(subjects) & set(subject_upper370))
        
            
        with open("rsfMRI_filtering_with_nan_sub_list.txt", mode="r") as file:
            nan_subjects = file.read().splitlines()
        subjects = set(subjects) - set(nan_subjects)
        
        if self.fine_tune_task == 'regression':
            cont_mean = non_na[self.target].mean()
            cont_std = non_na[self.target].std()
        
        for i,filename in enumerate(os.listdir(self.struct_dir)):
            # filename 형식: dti_count+smri_cortical_thickness_NDARINVZRHTXMXD.npy
            subject=filename.split('_')[-1].split('.')[0] # subject의 형식: NDARINVZRHTXMXD
            if subject in subjects:
                # Normalization
                if self.fine_tune_task == 'regression':
                    target = torch.tensor((self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0] - cont_mean) / cont_std)
                    target = target.float()
                elif self.fine_tune_task == 'binary_classification':
                    target = torch.tensor(self.meta_data.loc[self.meta_data['subjectkey']==subject,self.target].values[0]) 
                
                path_to_DTI_sMRIs = os.path.join(self.struct_dir, filename)
                path_to_fMRIs =  os.path.join(self.fmri_dir, 'sub-'+subject, 'desikankilliany_sub-'+subject+'.npy')
                self.index_l.append((i, subject, path_to_DTI_sMRIs, path_to_fMRIs, target))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_DTI_sMRIs, path_to_fMRIs, target = self.index_l[index]
        struct = np.load(path_to_DTI_sMRIs)
        struct = torch.Tensor(scipy.stats.zscore(struct, axis=None)).half()
        func = np.load(path_to_fMRIs)[20:].T # [84, 350 ~ 361]
        ts_length = func.shape[1]
        pad = 368-ts_length
        # frequency divide - padding ㄱㄱ
        T = TimeSeries(func, sampling_interval=0.8)
        FA = FilterAnalyzer(T, lb=0.0035)
        raw = scipy.stats.zscore(FA.data, axis=1)
        if self.filtering_type == 'FIR':
            low = scipy.stats.zscore(FA.fir.data, axis=1) #1) #(84, 353)
            ultralow = scipy.stats.zscore(FA.data-FA.fir.data, axis=1) #1) #(84, 353)
        elif self.filtering_type == 'Boxcar':
            low = scipy.stats.zscore(FA.filtered_boxcar.data, axis=1) #1) #(84, 353)
            ultralow = scipy.stats.zscore(FA.data-FA.filtered_boxcar.data, axis=1) #1) #(84, 353)

        # pad - raw                                
        raw = F.pad(torch.from_numpy(raw), (pad//2, pad-pad//2), "constant", 0) # (84, 368)
        raw = raw.T.float() #.type(torch.DoubleTensor)  # (368, 84)
            
        # pad - low                                
        low = F.pad(torch.from_numpy(low), (pad//2, pad-pad//2), "constant", 0) # (84, 368)
        low = low.T.float() #.type(torch.DoubleTensor)  # (368, 84)

        # pad - ultralow
        if self.feature_map_gen == 'resample' and self.feature_map_size == 'different':
            ultralow = AF.resample(waveform = torch.Tensor(ultralow),
                                   orig_freq = 3,
                                   new_freq = 1,
                                   resampling_method = 'sinc_interpolation') # torch.Size([84, 117])
            pad_ultralow = 128 - ultralow.shape[1]
            ultralow = F.pad(ultralow, (pad_ultralow//2, pad_ultralow-pad_ultralow//2), "constant", 0) # (84, 128) - pad on last dimension
        else:
            ultralow = F.pad(torch.from_numpy(ultralow), (pad//2, pad-pad//2), "constant", 0) # (84, 368)

        ultralow = ultralow.T.float() #.type(torch.DoubleTensor)  # (368, 84) or (128, 84)        
        
        ans_dict= {'fmri_raw_sequence':raw, 'fmri_lowfreq_sequence':low, 'fmri_ultralowfreq_sequence':ultralow, 'struct':struct, 'subject':subj,'subject_name':subj_name, self.target:target}
        
        return ans_dict
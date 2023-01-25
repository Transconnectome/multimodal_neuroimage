import gc
gc.collect()

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import model_selection
np.random.seed(0)

# 01 import data
raw_data = pd.read_csv('../../data/count.qc.csv') # DTI - count
raw_data['subjectkey'] = raw_data['subjectkey'].apply(lambda x : x.replace('_', ''))
raw_data = raw_data.drop_duplicates(subset='subjectkey', keep='first',inplace=False)

with open("../../multimodal_sub_list.txt", mode="r") as file:
    intersect = file.read().splitlines()
valid_sub = pd.DataFrame(intersect).rename(columns = {0:'subjectkey'})
manufactured_data = pd.merge(valid_sub, raw_data, how='inner', on='subjectkey')

phen = pd.read_csv('../../data/metadata/ABCD_phenotype_total.csv')

data = pd.merge(phen[['subjectkey', 'sex']], manufactured_data, how='inner', on='subjectkey')
data = data.dropna()
print(len(data))

start_subjectkey_index = np.where(data.columns.values == "subjectkey")[0][0]
start_meta_index = np.where(data.columns.values == "sex")[0][0] # Gender for hcp, sex for ABCD
start_brain_index = np.where(data.columns.values == "con_L.BSTS_L.CACG_count")[0][0]

subjectkey = list(data.columns[start_subjectkey_index:start_meta_index])
meta = list(data.columns[start_meta_index:start_brain_index])
brain = list(data.columns[start_brain_index:])

data = data[subjectkey+['sex']+brain]
data=data.dropna()
data = data.reset_index().iloc[:, 1:]

# 02 feature engineering
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train, test = train_test_split(data, test_size = 0.1, random_state = 27)
Num_FOLDS  = 5
# the number of feature that you want to show 
Num_feat = 20
target=['sex']

cv = model_selection.KFold(n_splits=Num_FOLDS, shuffle=True, random_state =0)
for train_index, test_index in cv.split(X=train, y=train[target]):
    print("TRAIN:", train_index, "TEST:", test_index)
    
    
for fold, (trn_, val_) in list(enumerate(cv.split(X=train, y=train[target]))):
    print(len(trn_), len(val_))
    
def preprocessing (train_data, test_data, NUM_FOLDS):
    test_data_processed= test_data.fillna(0).reset_index(drop=True)
    train_data_processed = train_data.fillna(0).reset_index(drop=True)
    
    # 초기값 설정
    test_data_processed["kfold"] = -1
    train_data_processed["kfold"] = -1

    # frac: 전체 row 중 몇 %를 반환할 지 결정 -> frac=1을 설정해서 모든 데이터를 반환
    # random_state: 추후 이것과 동일한 샘플링을 재현하기 위함
    # sample: 데이터에서 임의의 샘플 선정 -> frac=1이면 전체 data의 순서만 임의로 바뀜
    train_data_processed = train_data_processed.sample(frac=1,random_state=2021).reset_index(drop=True)

    # 5-fold cross validation --- regression은 CV 써야 함 target이 continuous라서...
    cv = model_selection.KFold(n_splits=NUM_FOLDS, shuffle=True, random_state =0)
    
    # enumerate: 각 split된 data set 순서대로 index를 함께 반환
    # ex. fold 0 일 때, 80%는 trn_, 20%는 val_
    
    # train data만 kfold 넣어줄거임.
    for fold, (trn_, val_) in enumerate(cv.split(X=train_data_processed, y=train_data_processed[target])):
        #print(len(trn_), len(val_)) -> 출력: 4개는 4071, 1018 / 1개는 4072, 1017
        # 'kfold' 칼럼은 cross validation할 때 fold 순서를 지정해놓은 것. 
        train_data_processed.loc[val_, 'kfold'] = fold
    
    print("done preprocessing")
    return train_data_processed, test_data_processed


train_data_processed, test_data_processed = preprocessing (train, test, Num_FOLDS)

def feature(Num_feat, clf_res, test_data_processed, features):
    importance_res = []
    for i in clf_res:
        importance_clf =i.feature_importances_
        importance_res.append(importance_clf)
    
    importance=[importance_res[0][i]/5+importance_res[1][i]/5+ importance_res[2][i]/5+ importance_res[3][i]/5+ importance_res[4][i]/5 for i in range(len(importance_res[1]))]
    

    #feat_name_sort=test_data_processed[features].columns[labels_importance]
    feat_name_sort = test_data_processed[features].columns
    important_features = pd.DataFrame([importance],columns = feat_name_sort, index=['Importance']) 
    important_features =important_features.transpose().sort_values(by=['Importance'], ascending=False)
    #important_features = important_features.head(50)

    return important_features


# Augmented
import itertools
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from tqdm.notebook import tqdm

def find_bestpar(fold, train_data_processed, features):
    
    # Store minimum auc
    max_auc = 0
    # Store maximum hypterparameter set
    max_hy = []
    
    # define hyperparameter space (quick version)
    m_ = [3, 6] #max_depth - 3, 4, 5, 6, 7, 8, 9, 10
    cw_ = [1, 7] #min_child_weight - 1, 3, 5, 7
    g_ = [0.0, 0.4] #gamma - 0.0, 0.1, 0.2 , 0.3, 0.4
    lr_ = [0.05, 0.30] #learning rate - 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
    cb_ = [0.6, 0.9] #colsample_bytree - 0.6, 0.7, 0.8 ,0.9 

    all_ = [m_, cw_, g_, lr_, cb_]
    h_space = [s for s in itertools.product(*all_)]
    
    for hy in tqdm(h_space):
        """===================Cross Validation==================="""
        
        """validation & valid 결과"""
        #valid_res = []
        valid_auc_res = []
        valid_bal_acc_res = []
     
        for i in range(fold):
            #print("fold ", i)
            # 5개의 fold 사용했으므로 변수 fold 값은 차례대로 0,1,2,3,4 중 하나
            clf = xgb.XGBClassifier(max_depth=hy[0], min_child_weight=hy[1],
                                    eval_metric='auc',
                                   gamma=hy[2], objective='binary:logistic', booster='gbtree',
                                   random_state=27, learning_rate=hy[3], colsample_bytree=hy[4],
                                   gpu_id=2, tree_method='gpu_hist', predictor='gpu_predictor',
                                   early_stopping_rounds=100, verbosity=0)

            df_train = train_data_processed[train_data_processed.kfold != i]  # 5개 중 4개 train에 할당
            df_valid = train_data_processed[train_data_processed.kfold == i]  # 5개 중 1개 validation에 할당
            
            X_train = df_train[features].values
            Y_train = df_train[target].values.reshape(-1,1)
            
            X_valid = df_valid[features].values
            Y_valid = df_valid[target].values.reshape(-1,1)
            

            clf.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_valid, Y_valid)])
                    #max_epochs=200 , patience=20, batch_size=1024, virtual_batch_size=128, num_workers=0, drop_last=False)

            preds = clf.predict(X_valid) # appears binary!!
            preds_proba = clf.predict_proba(X_valid)
            
            preds_proba = np.array(preds_proba).T[1]
            valid_bal_acc = metrics.balanced_accuracy_score(Y_valid, preds)
            valid_auc = metrics.roc_auc_score(Y_valid, preds_proba)
            
            #valid_res.append(clf.best_cost)
            valid_bal_acc_res.append(valid_bal_acc)
            valid_auc_res.append(valid_auc)

            print('[%3d/%4d] '%(i+1, fold),
                  'valid balanced accuracy: %.3f'%valid_bal_acc, 'valid AUC: %.3f'%valid_auc)
    

        """valid와 valid의 평균, 표준편차 출력"""
        print("=====parameter별 valid, valid score=====")
        print("valid balance accuracy 평균: %3f"%np.mean(valid_bal_acc_res),
              "valid AUC 평균: %3f"%np.mean(valid_auc_res))

        if np.mean(valid_auc_res)>max_auc:
            print("Find new maximum AUC!!")
            max_hy = hy
            max_auc = np.mean(valid_auc_res)
    
    return max_hy

def bestpar_tuning(fold, train_data_processed, test_data_processed, max_hy, features):
    hy = max_hy
    print("Max hy:" ,hy)
    X_test = test_data_processed[features].values
    Y_test = test_data_processed[target].values.reshape(-1,1)
    
    """validation & test 결과"""
    #valid_res = []
    test_auc_res = []
    test_bal_acc_res = []
    clf_res = []

    
    """해당 버전에 추가된 코드"""    
    y_valid_true = []
    y_valid_pred = []
    y_test_pred = []
    
    y_valid_subject = []
    y_test_subject = []
    
    
    for i in range(fold):
        clf = xgb.XGBClassifier(max_depth=hy[0], min_child_weight=hy[1],
                                    eval_metric='auc',
                                   gamma=hy[2], objective='binary:logistic', booster='gbtree',
                                   random_state=27, learning_rate=hy[3], colsample_bytree=hy[4],
                                   gpu_id=2, tree_method='gpu_hist', predictor='gpu_predictor',
                                   early_stopping_rounds=100, verbosity=0)
        
        # 5개의 fold 사용했으므로 변수 fold 값은 차례대로 0,1,2,3,4 중 하나
        df_train = train_data_processed[train_data_processed.kfold != i]  # 5개 중 4개 train에 할당
        df_valid = train_data_processed[train_data_processed.kfold == i]  # 5개 중 1개 validation에 할당
            
        X_train = df_train[features].values
        Y_train = df_train[target].values.reshape(-1, 1)
            
        X_valid = df_valid[features].values
        Y_valid = df_valid[target].values.reshape(-1, 1)
        
        #print(X_valid)
        # 학습 전 subject key 가져옴
        y_valid_subject.append(df_valid['subjectkey'].values)
        y_test_subject.append(test_data_processed['subjectkey'].values)  
        """해당 버전 추가 코드"""
        y_valid_true.append(Y_valid)        
        
        # 학습
        clf.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_valid, Y_valid)]) 
        
        # 결과들
        """fold별 validation & test data에 대한 target 예측 배열"""
        """해당 버전 추가 코드"""
        y_valid_pred.append(clf.predict(X_valid))
        y_test_pred.append(clf.predict(X_test))

        preds = clf.predict(X_test)
        preds_proba = clf.predict_proba(X_test)
        preds_proba = np.array(preds_proba).T[1]
        
        test_bal_acc = metrics.balanced_accuracy_score(Y_test, preds)
        test_auc = metrics.roc_auc_score(Y_test, preds_proba)

        #valid_res.append(clf.best_cost)
        test_bal_acc_res.append(test_bal_acc)
        test_auc_res.append(test_auc)

        print('[%3d/%4d] '%(i+1, fold),
              'Test balanced accuracy: %.3f'%test_bal_acc, 'Test AUC: %.3f'%test_auc)
        clf_res.append(clf)
    
    """valid와 test의 평균, 표준편차 출력"""
    print("Test balance accuracy 평균: %3f"%np.mean(test_bal_acc_res),
          "Test AUC 평균: %3f"%np.mean(test_auc_res))
    
    #valid_result = np.mean(valid_res)
    test_bal_acc = np.mean(test_bal_acc_res) 
    test_auc = np.mean(test_auc_res)
   
    return test_bal_acc, test_auc, clf_res, X_test, Y_test, y_valid_pred, y_test_pred, y_valid_subject, y_test_subject, y_valid_true


def run(train_data_processed, test_data_processed, fold, Num_feat, features):
    name_test = test_data_processed['subjectkey'].values
    print("-------------------------------Training Begining-------------------------------")
    m_ = [3, 6, 10] #max_depth - 3, 4, 5, 6, 7, 8, 9, 10
    cw_ = [1, 4, 7] #min_child_weight - 1, 3, 5, 7
    g_ = [0.0, 0.1, 0.4] #gamma - 0.0, 0.1, 0.2 , 0.3, 0.4
    lr_ = [0.05, 0.10, 0.30] #learning rate - 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
    cb_ = [0.6, 0.8, 0.9] #colsample_bytree - 0.6, 0.7, 0.8 ,0.9 
    
    all_ = [m_, cw_, g_, lr_, cb_]
    
    h_space = [s for s in itertools.product(*all_)]
    
    # Start training
    max_hy = find_bestpar(fold, train_data_processed, features)
    
    # if you want to just test the code, you should use this
    #max_hy = h_space[0]
    #print("Found maximum hyperparmeter, now work with that")
    
    print("-------------------------------Testing Begining-------------------------------")
    test_bal_acc, test_auc, clf_res, X_test, Y_test, y_valid_pred, y_test_pred, y_valid_subject, y_test_subject, y_valid_true = bestpar_tuning(fold, train_data_processed, test_data_processed, max_hy, features)
    
    #print("-------------------------------Important Feature-------------------------------")
    import_feat=feature(Num_feat, clf_res, test_data_processed, features)
    #import_feat=0
    #preds_val_prob = clf.predict_proba(X_valid)
    # important feature 다 나오는 코드임!!

    return test_bal_acc, test_auc, clf_res, X_test, Y_test, import_feat, name_test, y_valid_pred, y_test_pred, y_valid_subject, y_test_subject, y_valid_true


train_data_processed, test_data_processed = preprocessing (train, test, Num_FOLDS)

class model():
    def __init__(self, train_data_processed, test_data_processed, Num_FOLDS, Num_feat, features):
        test_bal_acc, test_auc, clf_res, X_test, Y_test, import_feat, name_test, y_valid_pred, y_test_pred, y_valid_subject, y_test_subject, y_valid_true = run(train_data_processed,test_data_processed,Num_FOLDS,Num_feat,features)
 
        self.train_data_processed = train_data_processed
        self.test_bal_acc = test_bal_acc
        self.test_auc = test_auc
        #self.valid_result = valid_result
        self.clf_res = clf_res  
        self.X_test = X_test
        self.Y_test = Y_test
        self.import_feat =  import_feat
        self.name_test = name_test
        self.features = features
        self.y_valid_pred =y_valid_pred
        self.y_test_pred = y_test_pred
        self.y_valid_subject= y_valid_subject
        self.y_test_subject = y_test_subject
        self.y_valid_true = y_valid_true
        
ROI_Xgboost = model(train_data_processed, test_data_processed, Num_FOLDS, Num_feat, brain)
print("ROI_Xgboost.done")
    

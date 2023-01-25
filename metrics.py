from sklearn.metrics import balanced_accuracy_score as bac
from sklearn.metrics import roc_curve,roc_auc_score, r2_score, f1_score, recall_score
import numpy as  np


class Metrics():
    def __init__(self):
        pass

    def BAC(self,truth,pred):
        return bac(truth,pred)

    def RAC(self,truth,pred):
        accuracy = np.sum([ii == jj for ii, jj in zip(truth, pred)]) / len(truth)
        return accuracy

    def AUROC(self,truth,pred):
        auroc_score = roc_auc_score(truth, pred)
        return auroc_score

    def ROC_CURVE(self,truth,pred,name,val_threshold):
        if name == 'test':
            print(f'using loaded threshold - {val_threshold} - for testing')
            best_threshold = val_threshold
            y_pred = [x>best_threshold for x in pred] # predicted label (0,1,1,0 ...)
            tpr = recall_score(truth, y_pred, pos_label=1)
            tnr = recall_score(truth, y_pred, pos_label=0)
            fpr = 1 - tnr
            best_gmean = np.sqrt(tpr * (1 - fpr))
            best_specificity = 1-fpr
            best_sensitivity = tpr
            best_f1_score = f1_score(truth, y_pred) 
            best_acc_score = self.BAC(truth, y_pred)
        else:
            print('calculating best thresholds with roc_curve')
            fpr, tpr, thresholds = roc_curve(truth, pred)
            # (1-fpr) denotes specificity
            # tpr denotes sensitivity (=recall)
            # calculate the g-mean for each threshold

            gmeans = np.sqrt(tpr * (1 - fpr))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)
            best_threshold = thresholds[ix]
            best_gmean = gmeans[ix]
            best_specificity = 1-fpr[ix]
            best_sensitivity = tpr[ix]
            best_f1_score = f1_score(truth, pred>best_threshold)
            best_acc_score = self.BAC(truth, pred>best_threshold)
        #print('Best Threshold={}, G-Mean={}, speicificity={}, sensitivity={}, f1_score={}'.format(best_threshold, gmeans[ix], 1-fpr[ix], tpr[ix], best_f1_score))
        return best_acc_score, best_threshold, best_gmean, best_specificity, best_sensitivity, best_f1_score

    def MAE(self,truth,pred):
        mae = np.mean([abs(ii - jj) for ii, jj in zip(pred, truth)])
        return mae

    def MSE(self,truth,pred):
        mse = np.mean([(ii - jj) ** 2 for ii, jj in zip(pred, truth)])
        return mse

    def NMSE(self,truth,pred):
        NMSE = np.mean([((ii - jj) ** 2) / (jj ** 2) for ii, jj in zip(pred, truth)])
        return NMSE
    
    def R2_score(self,truth,pred):
        R2 = r2_score(truth, pred)
        return R2
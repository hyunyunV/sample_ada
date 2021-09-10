# 데이터 조작
import pandas as pd
import numpy as np
import pyperclip
import math
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# 성과측정
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from scipy.stats import gmean
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# 모델 관련
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _check_sample_weight
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import optimizers
import tensorflow as tf

# 샘플링 관련
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
import collections
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE

class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled


    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))
    
def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def geoacc(mat):
    value = (mat.iloc[0,0] / sum(mat.iloc[0,:])) * (mat.iloc[1,1] / sum(mat.iloc[1,:]))
    return math.sqrt(value)

def calspe(mat) :
    value = (mat.iloc[1,1]/sum(mat.iloc[:,1]))
    return value

# clf를 estm 20개 배열로 받아서--> 이게 clf estimator를 20개로 처음부터 받는게아님 한개씩 훈련해야함 
# 내 생각엔 여기서 clfestimator를 넘겨주지말고 음.. X랑 y랑 던져줘서 객체에서 바로 훈련시키는게 나을 듯 

def makeconma(conma,mat):
    conma.iloc[0,0],conma.iloc[0,1],conma.iloc[1,0],conma.iloc[1,1]=mat[0,0],mat[0,1],mat[1,0],mat[1,1]
    return conma

def ToExcel(train_acc,train_geoacc,train_auc,train_spe, train_sen, test_acc,test_geoacc,test_auc,test_spe, test_sen, Ths): # ,mat,colname_first,colname_second
    mat = pd.DataFrame({"train_ACC":train_acc,"train_GEOACC":train_geoacc,"train_AUROC":train_auc,"train_SPE":train_spe,"train_SEN":train_sen,"test_ACC":test_acc,"test_GEOACC":test_geoacc,"test_AUROC":test_auc,"test_SPE":test_spe,"test_SEN":test_sen,"Ths":Ths})
    mat['Mean'] = mat.mean(axis=1)
    mat = mat.round(2)
    mat = mat.T
    return mat

def kfold_verify(lf, n_fold, X, y, name, n_estimators): 
    conma_1=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    conma_2=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    skfold = StratifiedKFold(n_splits=n_fold) # 이거는 10-fold니깐 10일 듯

    n_for = 4 # 이게 아마 3인가 그럼
    train_acc = []
    train_geoacc = []
    train_auc = []
    train_spe = []
    train_sen = []
    

    test_acc = []
    test_geoacc = []
    test_auc = []
    test_spe = []
    test_sen = []
    n_iter = 0
    n = 0
    
    Ths = []
    w = []
    conmat = []
    for i in range(n_for): # 여기서 3번
        #xlsx.sample(frac=1,random_state=i).reset_index(drop=True)
        for train_index, test_index in skfold.split(X,y): # 여기서 10 번 총 30번 이게 아마 skfold가 여기서 이렇게 선언하면 안댈텐데 나중에 만들때 고려해야함 아마 이 라인 바로위에서 skfold 객체를 만들어야할 것 
            clf = clone(lf)    
            idx_li = ['fold','train/test','ratio1','ratio2','ratio3','ratio4','ratio5','ratio6','ratio7','target','predict','total_proba']
            frame = pd.DataFrame()
            n_iter +=1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            #---데이터 정규화
            X_train_s = X_train
            X_test_s = X_test
            #

            
            clf.fit(X_train_s,y_train)             
            
            N = clf.count
            
            #----------------------SAMME, SAMME.R 자꾸 햇갈려서 걍 전처리
            clf.estimator_weights_ = clf.estimator_weights_/sum(clf.estimator_weights_)
            proba1 = 0
            for i in range(N):
                proba1 += clf.estimators_[i].predict_proba(X_train_s)*clf.estimator_weights_[i]
            proba1 = proba1[:,1]
            proba2 = 0
            for i in range(N):
                proba2 += clf.estimators_[i].predict_proba(X_test_s)*clf.estimator_weights_[i]
            proba2 = proba2[:,1]
            #----------------------

            #------------------------------- 임계값 구하기
            Threshold = Find_Optimal_Cutoff(y_train, proba1)
            Ths.append(Threshold)
            
            train_pred = proba1 >= Threshold
            pred = proba2 >= Threshold
            
            mat_train  = confusion_matrix(y_train,train_pred)
            mat_test  = confusion_matrix(y_test,pred)
            
            conma_train = makeconma(conma_1,mat_train)
            conma_test = makeconma(conma_2,mat_test)            
            
            conmat.append(np.concatenate([mat_train,mat_test],axis = 0))
            w.append(clf.estimator_weights_) 
            #------------------------------- 임계값 구하기
            
            #------------------------------- 성능 측정 train
            accuracy = np.round(accuracy_score(y_train,train_pred), 4 )
            auc = roc_auc_score(y_train, proba1)
            geoaccvalue = geoacc(conma_train)
            spe = calspe(conma_train)
            sen = recall_score(y_train,train_pred)
            
            train_acc.append(accuracy)
            train_geoacc.append(geoaccvalue)
            train_auc.append(auc)            
            train_spe.append(spe)
            train_sen.append(sen)
            
            #------------------------------- 성능 측정
            
            
            #------------------------------- 성능 측정 test
            accuracy = np.round(accuracy_score(y_test,pred), 4 )
            auc = roc_auc_score(y_test, proba2)
            geoaccvalue = geoacc(conma_test)
            spe = calspe(conma_test)
            sen = recall_score(y_test,pred)

            test_acc.append(accuracy)
            test_geoacc.append(geoaccvalue)
            test_auc.append(auc)            
            test_spe.append(spe)            
            test_sen.append(sen)            
            
            #------------------------------- 성능 측정
            
            #------------------------------- 수정해야댈 듯
            # each 20 estm predict
            train_proba = []
            train_pred = proba1 >Threshold
            for j in range(N):
                train_proba.append(clf.estimators_[j].predict_proba(X_train_s)) 

            test_proba = []
            test_pred = 0
            
            test_pred = proba2 >Threshold
            for j in range(N):
                test_proba.append(clf.estimators_[j].predict_proba(X_test_s))
            #-------------------------------------
            X_train_s = np.array(X_train_s)
            y_train = np.array(y_train)

            X_test = np.array(X_test)
            y_test = np.array(y_test)
            for i in range(len(train_proba[0])):
                li = list(X_train_s[i])
                li.insert(0,'train')
                li.insert(0,(n_iter));
                li.append(y_train[i])
                li.append(train_pred[i])
                li.append(proba1[i])
                for j in range(N): # 여기서 20개 append
                    li.append(train_proba[j][i][0]) # proba 0 추가 약분류기 j 번째에서 
                    li.append(train_proba[j][i][1]) # proba 1추가
                frame = frame.append(pd.Series(li,index=idx_li),ignore_index=True)
            
            for i in range(len(test_proba[0])):
                li = list(X_test[i])
                li.insert(0,'test')
                li.insert(0,(n_iter));
                li.append(y_test[i])
                li.append(test_pred[i])
                li.append(proba2[i])
                for j in range(N): # 여기서 20개 append
                    li.append(test_proba[j][i][0]) # proba 0 추가
                    li.append(test_proba[j][i][1]) # proba 1 추가
                frame = frame.append(pd.Series(li),ignore_index=True)
         
            for i in range(N):
                idx_li.append('estm '+str(i)+' proba 0')
                idx_li.append('estm '+str(i)+' proba 1')
            frame.columns = idx_li
            frame.to_csv("ResultRUSROS/"+name+str(n)+"-th_"+str(len(X))+'_input.csv')
            n = n + 1
            if n == 30:
                break
    result = ToExcel(train_acc, train_geoacc, train_auc, train_spe, train_sen, test_acc, test_geoacc, test_auc, test_spe, test_sen, Ths)
    result.to_csv("ResultRUSROS/"+name+"_result"+str(len(X))+".csv")
    len_w = w[0].shape[0]
    w = pd.DataFrame(np.concatenate(w).reshape(-1, len_w))
    w.to_csv("ResultRUSROS/weight"+name+str(len(X))+".csv")    
    conmat = pd.DataFrame(np.concatenate(conmat,axis=1))
    conmat.to_csv("ResultRUSROS/conmat"+name+str(len(X))+".csv")

def kfold_verify_RUS(lf, n_fold, X, y, name, n_estimators): 
    conma_1=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    conma_2=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    skfold = StratifiedKFold(n_splits=n_fold) # 이거는 10-fold니깐 10일 듯

    n_for = 4 # 이게 아마 3인가 그럼
    train_acc = []
    train_geoacc = []
    train_auc = []
    train_spe = []
    train_sen = []
    

    test_acc = []
    test_geoacc = []
    test_auc = []
    test_spe = []
    test_sen = []
    n_iter = 0
    n = 0
    
    Ths = []
    w = []
    conmat = []
    for i in range(n_for): # 여기서 3번
        #xlsx.sample(frac=1,random_state=i).reset_index(drop=True)
        for train_index, test_index in skfold.split(X,y): # 여기서 10 번 총 30번 이게 아마 skfold가 여기서 이렇게 선언하면 안댈텐데 나중에 만들때 고려해야함 아마 이 라인 바로위에서 skfold 객체를 만들어야할 것 
            clf = clone(lf)    
            idx_li = ['fold','train/test','ratio1','ratio2','ratio3','ratio4','ratio5','ratio6','ratio7','target','predict','total_proba']
            frame = pd.DataFrame()
            n_iter +=1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            #---데이터 정규화
            X_train_s = X_train
            X_test_s = X_test
            #

            #--RUS 적용
            under = RandomUnderSampler(sampling_strategy = 1)
            X_train_r, y_train_r = under.fit_resample(X_train_s,y_train)
            try :
                clf.fit(X_train_r,y_train_r)             
            except :
                continue
            N = clf.count

            
            #----------------------SAMME, SAMME.R 자꾸 햇갈려서 걍 전처리
            clf.estimator_weights_ = clf.estimator_weights_/sum(clf.estimator_weights_)
            proba1 = 0
            for i in range(N):
                proba1 += clf.estimators_[i].predict_proba(X_train_r)*clf.estimator_weights_[i]
            proba1 = proba1[:,1]
            proba2 = 0
            for i in range(N):
                proba2 += clf.estimators_[i].predict_proba(X_test_s)*clf.estimator_weights_[i]
            proba2 = proba2[:,1]
            #----------------------

            #------------------------------- 임계값 구하기
            Threshold = Find_Optimal_Cutoff(y_train_r, proba1)
            Ths.append(Threshold)
            
            train_pred = proba1 >= Threshold
            pred = proba2 >= Threshold
            
            mat_train  = confusion_matrix(y_train_r,train_pred)
            mat_test  = confusion_matrix(y_test,pred)
            
            conma_train = makeconma(conma_1,mat_train)
            conma_test = makeconma(conma_2,mat_test)            
            
            conmat.append(np.concatenate([mat_train,mat_test],axis = 0))
            w.append(clf.estimator_weights_) 
            #------------------------------- 임계값 구하기
            
            #------------------------------- 성능 측정 train
            accuracy = np.round(accuracy_score(y_train_r,train_pred), 4 )
            auc = roc_auc_score(y_train_r, proba1)
            geoaccvalue = geoacc(conma_train)
            spe = calspe(conma_train)
            sen = recall_score(y_train_r,train_pred)
            
            train_acc.append(accuracy)
            train_geoacc.append(geoaccvalue)
            train_auc.append(auc)            
            train_spe.append(spe)
            train_sen.append(sen)
            
            #------------------------------- 성능 측정
            
            
            #------------------------------- 성능 측정 test
            accuracy = np.round(accuracy_score(y_test,pred), 4 )
            auc = roc_auc_score(y_test, proba2)
            geoaccvalue = geoacc(conma_test)
            spe = calspe(conma_test)
            sen = recall_score(y_test,pred)

            test_acc.append(accuracy)
            test_geoacc.append(geoaccvalue)
            test_auc.append(auc)            
            test_spe.append(spe)            
            test_sen.append(sen)            
            
            #------------------------------- 성능 측정
            
            #------------------------------- 수정해야댈 듯
            # each 20 estm predict
            train_proba = []
            train_pred = proba1 >Threshold
            for j in range(N):
                train_proba.append(clf.estimators_[j].predict_proba(X_train_r)) 

            test_proba = []
            test_pred = 0
            
            test_pred = proba2 >Threshold
            for j in range(N):
                test_proba.append(clf.estimators_[j].predict_proba(X_test_s))
            #-------------------------------------
            X_train_r = np.array(X_train_r)
            y_train_r = np.array(y_train_r)

            X_test = np.array(X_test)
            y_test = np.array(y_test)
            for i in range(len(train_proba[0])):
                li = list(X_train_r[i])
                li.insert(0,'train')
                li.insert(0,(n_iter));
                li.append(y_train_r[i])
                li.append(train_pred[i])
                li.append(proba1[i])
                for j in range(N): # 여기서 20개 append
                    li.append(train_proba[j][i][0]) # proba 0 추가 약분류기 j 번째에서 
                    li.append(train_proba[j][i][1]) # proba 1추가
                frame = frame.append(pd.Series(li),ignore_index=True)
            
            for i in range(len(test_proba[0])):
                li = list(X_test[i])
                li.insert(0,'test')
                li.insert(0,(n_iter));
                li.append(y_test[i])
                li.append(test_pred[i])
                li.append(proba2[i])
                for j in range(N): # 여기서 20개 append
                    li.append(test_proba[j][i][0]) # proba 0 추가
                    li.append(test_proba[j][i][1]) # proba 1 추가
                frame = frame.append(pd.Series(li),ignore_index=True)

            for i in range(N):
                idx_li.append('estm '+str(i)+' proba 0')
                idx_li.append('estm '+str(i)+' proba 1')
            frame.columns = idx_li
            frame.to_csv("ResultRUSROS/"+name+str(n)+"-th_"+str(len(X))+'_input.csv')
            n = n + 1
            if n == 30:
                break
    result = ToExcel(train_acc, train_geoacc, train_auc, train_spe, train_sen, test_acc, test_geoacc, test_auc, test_spe, test_sen, Ths)
    result.to_csv("ResultRUSROS/"+name+"_result"+str(len(X))+".csv")
    len_w = w[0].shape[0]
    w = pd.DataFrame(np.concatenate(w).reshape(-1, len_w))
    w.to_csv("ResultRUSROS/weight"+name+str(len(X))+".csv")    
    conmat = pd.DataFrame(np.concatenate(conmat,axis=1))
    conmat.to_csv("ResultRUSROS/conmat"+name+str(len(X))+".csv")

def kfold_verify_CUS(lf, n_fold, X, y, name, n_estimators): 
    conma_1=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    conma_2=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    skfold = StratifiedKFold(n_splits=n_fold) # 이거는 10-fold니깐 10일 듯

    n_for = 4 # 이게 아마 3인가 그럼
    train_acc = []
    train_geoacc = []
    train_auc = []
    train_spe = []
    train_sen = []
    

    test_acc = []
    test_geoacc = []
    test_auc = []
    test_spe = []
    test_sen = []
    n_iter = 0
    n = 0
    
    Ths = []
    w = []
    conmat = []
    for i in range(n_for): # 여기서 3번
        #xlsx.sample(frac=1,random_state=i).reset_index(drop=True)
        for train_index, test_index in skfold.split(X,y): # 여기서 10 번 총 30번 이게 아마 skfold가 여기서 이렇게 선언하면 안댈텐데 나중에 만들때 고려해야함 아마 이 라인 바로위에서 skfold 객체를 만들어야할 것 
            clf = clone(lf)    
            idx_li = ['fold','train/test','ratio1','ratio2','ratio3','ratio4','ratio5','ratio6','ratio7','target','predict','total_proba']
            frame = pd.DataFrame()
            n_iter +=1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            #---데이터 정규화
            X_train_s = X_train
            X_test_s = X_test
            #

            #--RUS 적용
            under = ClusterCentroids(sampling_strategy=1,voting = 'hard')
            X_train_r, y_train_r = under.fit_resample(X_train_s,y_train)
            
            try :
                clf.fit(X_train_r,y_train_r)             
            except :
                continue 
            
            N = clf.count


            
            #----------------------SAMME, SAMME.R 자꾸 햇갈려서 걍 전처리
            clf.estimator_weights_ = clf.estimator_weights_/sum(clf.estimator_weights_)
            proba1 = 0
            for i in range(N):
                proba1 += clf.estimators_[i].predict_proba(X_train_r)*clf.estimator_weights_[i]
            proba1 = proba1[:,1]
            proba2 = 0
            for i in range(N):
                proba2 += clf.estimators_[i].predict_proba(X_test_s)*clf.estimator_weights_[i]
            proba2 = proba2[:,1]
            #----------------------

            #------------------------------- 임계값 구하기
            Threshold = Find_Optimal_Cutoff(y_train_r, proba1)
            Ths.append(Threshold)
            
            train_pred = proba1 >= Threshold
            pred = proba2 >= Threshold
            
            mat_train  = confusion_matrix(y_train_r,train_pred)
            mat_test  = confusion_matrix(y_test,pred)
            
            conma_train = makeconma(conma_1,mat_train)
            conma_test = makeconma(conma_2,mat_test)            
            
            conmat.append(np.concatenate([mat_train,mat_test],axis = 0))
            w.append(clf.estimator_weights_) 
            #------------------------------- 임계값 구하기
            
            #------------------------------- 성능 측정 train
            accuracy = np.round(accuracy_score(y_train_r,train_pred), 4 )
            auc = roc_auc_score(y_train_r, proba1)
            geoaccvalue = geoacc(conma_train)
            spe = calspe(conma_train)
            sen = recall_score(y_train_r,train_pred)
            
            train_acc.append(accuracy)
            train_geoacc.append(geoaccvalue)
            train_auc.append(auc)            
            train_spe.append(spe)
            train_sen.append(sen)
            
            #------------------------------- 성능 측정
            
            
            #------------------------------- 성능 측정 test
            accuracy = np.round(accuracy_score(y_test,pred), 4 )
            auc = roc_auc_score(y_test, proba2)
            geoaccvalue = geoacc(conma_test)
            spe = calspe(conma_test)
            sen = recall_score(y_test,pred)

            test_acc.append(accuracy)
            test_geoacc.append(geoaccvalue)
            test_auc.append(auc)            
            test_spe.append(spe)            
            test_sen.append(sen)            
            
            #------------------------------- 성능 측정
            
            #------------------------------- 수정해야댈 듯
            # each 20 estm predict
            train_proba = []
            train_pred = proba1 >Threshold
            for j in range(N):
                train_proba.append(clf.estimators_[j].predict_proba(X_train_r)) 

            test_proba = []
            test_pred = 0
            
            test_pred = proba2 >Threshold
            for j in range(N):
                test_proba.append(clf.estimators_[j].predict_proba(X_test_s))
            #-------------------------------------
            X_train_r = np.array(X_train_r)
            y_train_r = np.array(y_train_r)

            X_test = np.array(X_test)
            y_test = np.array(y_test)
            for i in range(len(train_proba[0])):
                li = list(X_train_r[i])
                li.insert(0,'train')
                li.insert(0,(n_iter));
                li.append(y_train_r[i])
                li.append(train_pred[i])
                li.append(proba1[i])
                for j in range(N): # 여기서 20개 append
                    li.append(train_proba[j][i][0]) # proba 0 추가 약분류기 j 번째에서 
                    li.append(train_proba[j][i][1]) # proba 1추가
                frame = frame.append(pd.Series(li),ignore_index=True)
            
            for i in range(len(test_proba[0])):
                li = list(X_test[i])
                li.insert(0,'test')
                li.insert(0,(n_iter));
                li.append(y_test[i])
                li.append(test_pred[i])
                li.append(proba2[i])
                for j in range(N): # 여기서 20개 append
                    li.append(test_proba[j][i][0]) # proba 0 추가
                    li.append(test_proba[j][i][1]) # proba 1 추가
                frame = frame.append(pd.Series(li),ignore_index=True)
            
            for i in range(N):
                idx_li.append('estm '+str(i)+' proba 0')
                idx_li.append('estm '+str(i)+' proba 1')
            frame.columns = idx_li
            frame.to_csv("ResultRUSROS/"+name+str(n)+"-th_"+str(len(X))+'_input.csv')
            n = n + 1
            if n == 30:
                break
    result = ToExcel(train_acc, train_geoacc, train_auc, train_spe, train_sen, test_acc, test_geoacc, test_auc, test_spe, test_sen, Ths)
    result.to_csv("ResultRUSROS/"+name+"_result"+str(len(X))+".csv")
    len_w = w[0].shape[0]
    w = pd.DataFrame(np.concatenate(w).reshape(-1, len_w))
    w.to_csv("ResultRUSROS/weight"+name+str(len(X))+".csv")    
    conmat = pd.DataFrame(np.concatenate(conmat,axis=1))
    conmat.to_csv("ResultRUSROS/conmat"+name+str(len(X))+".csv")

def kfold_verify_ROS(lf, n_fold, X, y, name, n_estimators): 
    conma_1=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    conma_2=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    skfold = StratifiedKFold(n_splits=n_fold) # 이거는 10-fold니깐 10일 듯

    n_for = 4 # 이게 아마 3인가 그럼
    train_acc = []
    train_geoacc = []
    train_auc = []
    train_spe = []
    train_sen = []
    

    test_acc = []
    test_geoacc = []
    test_auc = []
    test_spe = []
    test_sen = []
    n_iter = 0
    n = 0
    
    Ths = []
    w = []
    conmat = []
    for i in range(n_for): # 여기서 3번
        #xlsx.sample(frac=1,random_state=i).reset_index(drop=True)
        for train_index, test_index in skfold.split(X,y): # 여기서 10 번 총 30번 이게 아마 skfold가 여기서 이렇게 선언하면 안댈텐데 나중에 만들때 고려해야함 아마 이 라인 바로위에서 skfold 객체를 만들어야할 것 
            clf = clone(lf)    
            idx_li = ['fold','train/test','ratio1','ratio2','ratio3','ratio4','ratio5','ratio6','ratio7','target','predict','total_proba']
            frame = pd.DataFrame()
            n_iter +=1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            #---데이터 정규화
            X_train_s = X_train
            X_test_s = X_test
            #

            #--RUS 적용
            ros = RandomOverSampler(random_state=0)
            X_train_r, y_train_r = ros.fit_resample(X_train_s, y_train)
            try :
                clf.fit(X_train_r,y_train_r)             
            except:
                continue 
            N = clf.count

            
            #----------------------SAMME, SAMME.R 자꾸 햇갈려서 걍 전처리
            clf.estimator_weights_ = clf.estimator_weights_/sum(clf.estimator_weights_)
            proba1 = 0
            for i in range(N):
                proba1 += clf.estimators_[i].predict_proba(X_train_r)*clf.estimator_weights_[i]
            proba1 = proba1[:,1]
            proba2 = 0
            for i in range(N):
                proba2 += clf.estimators_[i].predict_proba(X_test_s)*clf.estimator_weights_[i]
            proba2 = proba2[:,1]
            #----------------------

            #------------------------------- 임계값 구하기
            Threshold = Find_Optimal_Cutoff(y_train_r, proba1)
            Ths.append(Threshold)
            
            train_pred = proba1 >= Threshold
            pred = proba2 >= Threshold
            
            mat_train  = confusion_matrix(y_train_r,train_pred)
            mat_test  = confusion_matrix(y_test,pred)
            
            conma_train = makeconma(conma_1,mat_train)
            conma_test = makeconma(conma_2,mat_test)            
            
            conmat.append(np.concatenate([mat_train,mat_test],axis = 0))
            w.append(clf.estimator_weights_) 
            #------------------------------- 임계값 구하기
            
            #------------------------------- 성능 측정 train
            accuracy = np.round(accuracy_score(y_train_r,train_pred), 4 )
            auc = roc_auc_score(y_train_r, proba1)
            geoaccvalue = geoacc(conma_train)
            spe = calspe(conma_train)
            sen = recall_score(y_train_r,train_pred)
            
            train_acc.append(accuracy)
            train_geoacc.append(geoaccvalue)
            train_auc.append(auc)            
            train_spe.append(spe)
            train_sen.append(sen)
            
            #------------------------------- 성능 측정
            
            
            #------------------------------- 성능 측정 test
            accuracy = np.round(accuracy_score(y_test,pred), 4 )
            auc = roc_auc_score(y_test, proba2)
            geoaccvalue = geoacc(conma_test)
            spe = calspe(conma_test)
            sen = recall_score(y_test,pred)

            test_acc.append(accuracy)
            test_geoacc.append(geoaccvalue)
            test_auc.append(auc)            
            test_spe.append(spe)            
            test_sen.append(sen)            
            
            #------------------------------- 성능 측정
            
            #------------------------------- 수정해야댈 듯
            # each 20 estm predict
            train_proba = []
            train_pred = proba1 >Threshold
            for j in range(N):
                train_proba.append(clf.estimators_[j].predict_proba(X_train_r)) 

            test_proba = []
            test_pred = 0
            
            test_pred = proba2 >Threshold
            for j in range(N):
                test_proba.append(clf.estimators_[j].predict_proba(X_test_s))
            #-------------------------------------
            X_train_s = np.array(X_train_r)
            y_train = np.array(y_train_r)

            X_test = np.array(X_test)
            y_test = np.array(y_test)
            for i in range(len(train_proba[0])):
                li = list(X_train_s[i])
                li.insert(0,'train')
                li.insert(0,(n_iter));
                li.append(y_train[i])
                li.append(train_pred[i])
                li.append(proba1[i])
                for j in range(N): # 여기서 20개 append
                    li.append(train_proba[j][i][0]) # proba 0 추가 약분류기 j 번째에서 
                    li.append(train_proba[j][i][1]) # proba 1추가
                frame = frame.append(pd.Series(li),ignore_index=True)
            
            for i in range(len(test_proba[0])):
                li = list(X_test[i])
                li.insert(0,'test')
                li.insert(0,(n_iter));
                li.append(y_test[i])
                li.append(test_pred[i])
                li.append(proba2[i])
                for j in range(N): # 여기서 20개 append
                    li.append(test_proba[j][i][0]) # proba 0 추가
                    li.append(test_proba[j][i][1]) # proba 1 추가
                frame = frame.append(pd.Series(li),ignore_index=True)

            for i in range(N):
                idx_li.append('estm '+str(i)+' proba 0')
                idx_li.append('estm '+str(i)+' proba 1')
            frame.columns = idx_li
            frame.to_csv("ResultRUSROS/"+name+str(n)+"-th_"+str(len(X))+'_input.csv')
            n = n + 1
            if n == 30:
                break
    result = ToExcel(train_acc, train_geoacc, train_auc, train_spe, train_sen, test_acc, test_geoacc, test_auc, test_spe, test_sen, Ths)
    result.to_csv("ResultRUSROS/"+name+"_result"+str(len(X))+".csv")
    len_w = w[0].shape[0]
    w = pd.DataFrame(np.concatenate(w).reshape(-1, len_w))
    w.to_csv("ResultRUSROS/weight"+name+str(len(X))+".csv")    
    conmat = pd.DataFrame(np.concatenate(conmat,axis=1))
    conmat.to_csv("ResultRUSROS/conmat"+name+str(len(X))+".csv")

def kfold_verify_SMO(lf, n_fold, X, y, name, n_estimators): 
    conma_1=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    conma_2=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    skfold = StratifiedKFold(n_splits=n_fold) # 이거는 10-fold니깐 10일 듯

    n_for = 4 # 이게 아마 3인가 그럼
    train_acc = []
    train_geoacc = []
    train_auc = []
    train_spe = []
    train_sen = []
    

    test_acc = []
    test_geoacc = []
    test_auc = []
    test_spe = []
    test_sen = []
    n_iter = 0
    n = 0
    
    Ths = []
    w = []
    conmat = []
    for i in range(n_for): # 여기서 3번
        #xlsx.sample(frac=1,random_state=i).reset_index(drop=True)
        for train_index, test_index in skfold.split(X,y): # 여기서 10 번 총 30번 이게 아마 skfold가 여기서 이렇게 선언하면 안댈텐데 나중에 만들때 고려해야함 아마 이 라인 바로위에서 skfold 객체를 만들어야할 것 
            clf = clone(lf)    
            idx_li = ['fold','train/test','ratio1','ratio2','ratio3','ratio4','ratio5','ratio6','ratio7','target','predict','total_proba']
            frame = pd.DataFrame()
            n_iter +=1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            #---데이터 정규화
            X_train_s = X_train
            X_test_s = X_test
            #

            #--RUS 적용
            sm = SMOTE(sampling_strategy = 1)
            X_train_r, y_train_r = sm.fit_resample(X_train_s, y_train)
            try :
                clf.fit(X_train_r,y_train_r)             
            except:
                continue 
            N = clf.count

            
            #----------------------SAMME, SAMME.R 자꾸 햇갈려서 걍 전처리
            clf.estimator_weights_ = clf.estimator_weights_/sum(clf.estimator_weights_)
            proba1 = 0
            for i in range(N):
                proba1 += clf.estimators_[i].predict_proba(X_train_r)*clf.estimator_weights_[i]
            proba1 = proba1[:,1]
            proba2 = 0
            for i in range(N):
                proba2 += clf.estimators_[i].predict_proba(X_test_s)*clf.estimator_weights_[i]
            proba2 = proba2[:,1]
            #----------------------

            #------------------------------- 임계값 구하기
            Threshold = Find_Optimal_Cutoff(y_train_r, proba1)
            Ths.append(Threshold)
            
            train_pred = proba1 >= Threshold
            pred = proba2 >= Threshold
            
            mat_train  = confusion_matrix(y_train_r,train_pred)
            mat_test  = confusion_matrix(y_test,pred)
            
            conma_train = makeconma(conma_1,mat_train)
            conma_test = makeconma(conma_2,mat_test)            
            
            conmat.append(np.concatenate([mat_train,mat_test],axis = 0))
            w.append(clf.estimator_weights_) 
            #------------------------------- 임계값 구하기
            
            #------------------------------- 성능 측정 train
            accuracy = np.round(accuracy_score(y_train_r,train_pred), 4 )
            auc = roc_auc_score(y_train_r, proba1)
            geoaccvalue = geoacc(conma_train)
            spe = calspe(conma_train)
            sen = recall_score(y_train_r,train_pred)
            
            train_acc.append(accuracy)
            train_geoacc.append(geoaccvalue)
            train_auc.append(auc)            
            train_spe.append(spe)
            train_sen.append(sen)
            
            #------------------------------- 성능 측정
            
            
            #------------------------------- 성능 측정 test
            accuracy = np.round(accuracy_score(y_test,pred), 4 )
            auc = roc_auc_score(y_test, proba2)
            geoaccvalue = geoacc(conma_test)
            spe = calspe(conma_test)
            sen = recall_score(y_test,pred)

            test_acc.append(accuracy)
            test_geoacc.append(geoaccvalue)
            test_auc.append(auc)            
            test_spe.append(spe)            
            test_sen.append(sen)            
            
            #------------------------------- 성능 측정
            
            #------------------------------- 수정해야댈 듯
            # each 20 estm predict
            train_proba = []
            train_pred = proba1 >Threshold
            for j in range(N):
                train_proba.append(clf.estimators_[j].predict_proba(X_train_r)) 

            test_proba = []
            test_pred = 0
            
            test_pred = proba2 >Threshold
            for j in range(N):
                test_proba.append(clf.estimators_[j].predict_proba(X_test_s))
            #-------------------------------------
            X_train_s = np.array(X_train_r)
            y_train = np.array(y_train_r)

            X_test = np.array(X_test)
            y_test = np.array(y_test)
            for i in range(len(train_proba[0])):
                li = list(X_train_s[i])
                li.insert(0,'train')
                li.insert(0,(n_iter));
                li.append(y_train[i])
                li.append(train_pred[i])
                li.append(proba1[i])
                for j in range(N): # 여기서 20개 append
                    li.append(train_proba[j][i][0]) # proba 0 추가 약분류기 j 번째에서 
                    li.append(train_proba[j][i][1]) # proba 1추가
                frame = frame.append(pd.Series(li),ignore_index=True)
            
            for i in range(len(test_proba[0])):
                li = list(X_test[i])
                li.insert(0,'test')
                li.insert(0,(n_iter));
                li.append(y_test[i])
                li.append(test_pred[i])
                li.append(proba2[i])
                for j in range(N): # 여기서 20개 append
                    li.append(test_proba[j][i][0]) # proba 0 추가
                    li.append(test_proba[j][i][1]) # proba 1 추가
                frame = frame.append(pd.Series(li),ignore_index=True)

            for i in range(N):
                idx_li.append('estm '+str(i)+' proba 0')
                idx_li.append('estm '+str(i)+' proba 1')
            frame.columns = idx_li
            frame.to_csv("ResultRUSROS/"+name+str(n)+"-th_"+str(len(X))+'_input.csv')
            n = n + 1
            if n == 30:
                break
    result = ToExcel(train_acc, train_geoacc, train_auc, train_spe, train_sen, test_acc, test_geoacc, test_auc, test_spe, test_sen, Ths)
    result.to_csv("ResultRUSROS/"+name+"_result"+str(len(X))+".csv")
    len_w = w[0].shape[0]
    w = pd.DataFrame(np.concatenate(w).reshape(-1, len_w))
    w.to_csv("ResultRUSROS/weight"+name+str(len(X))+".csv")    
    conmat = pd.DataFrame(np.concatenate(conmat,axis=1))
    conmat.to_csv("ResultRUSROS/conmat"+name+str(len(X))+".csv")

def fold1(clf, xlsx, X , y, name): 
    idx_li = ['fold','train/test','ratio1','ratio2','ratio3','ratio4','ratio5','ratio6','ratio7','target','predict']
    conma_1=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])
    conma_2=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])

    train_acc = []
    train_geoacc = []
    train_auc = []

    test_acc = []
    test_geoacc = []
    test_auc = []
    n_iter = 0
    n = 0
    
    Ths = []
    w = []
    conmat = []
    
    #xlsx.sample(frac=1,random_state=i).reset_index(drop=True)
    X = xlsx.iloc[:,:7]
    y = xlsx.iloc[:,7]
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify = y)
    frame = pd.DataFrame(columns = idx_li)
    n_iter +=1
    ## X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    ## y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train,y_train)             
    N = len(clf.estimators_)
    for i in range(N):
        idx_li.append('estm '+str(i)+' proba 0')
        idx_li.append('estm '+str(i)+' proba 1')
    
    #----------------------SAMME, SAMME.R 자꾸 햇갈려서 걍 전처리
    clf.estimator_weights_ = clf.estimator_weights_/sum(clf.estimator_weights_)
    proba1 = 0
    for i in range(len(clf.estimators_)):
        proba1 += clf.estimators_[i].predict_proba(X_train)*clf.estimator_weights_[i]
    proba1 = proba1[:,1]
    proba2 = 0
    for i in range(len(clf.estimators_)):
        proba2 += clf.estimators_[i].predict_proba(X_test)*clf.estimator_weights_[i]
    proba2 = proba2[:,1]
    #----------------------
        
    
    #------------------------------- 임계값 구하기
    Threshold = Find_Optimal_Cutoff(y_train, proba1)
    Ths.append(Threshold)

    train_pred = proba1 >= Threshold
    pred = proba2 >= Threshold
    
    mat_train  = confusion_matrix(y_train,train_pred)
    mat_test  = confusion_matrix(y_test,pred)
    
    conma_train = makeconma(conma_1,mat_train)
    conma_test = makeconma(conma_2,mat_test)            
    
    conmat.append(np.concatenate([mat_train,mat_test],axis = 0))
    w.append(clf.estimator_weights_/sum(clf.estimator_weights_)) # 이게 데이터 프레임 형태가 어떤지 모르겠네

    #------------------------------- 임계값 구하기
    
    #------------------------------- 성능 측정 train
    accuracy = np.round(accuracy_score(y_train,train_pred), 4 )
    auc = roc_auc_score(y_train, proba1)
    geoaccvalue = geoacc(conma_train)
    train_acc.append(accuracy)
    train_geoacc.append(geoaccvalue)
    train_auc.append(auc)            
    
    #------------------------------- 성능 측정
    
    
    #------------------------------- 성능 측정 test
    accuracy = np.round(accuracy_score(y_test,pred), 4 )
    auc = roc_auc_score(y_test, proba2)
    geoaccvalue = geoacc(conma_test)
    test_acc.append(accuracy)
    test_geoacc.append(geoaccvalue)
    test_auc.append(auc)            
    
    #------------------------------- 성능 측정
    #------------------------------- 수정해야댈 듯
    # each 20 estm predict
    train_proba = []
    train_pred = 0
    for j in range(N):
        train_proba.append(clf.estimators_[j].predict_proba(X_train)) 
        train_pred = clf.predict(X_train)

    test_proba = []
    test_pred = 0

    for j in range(N):
        test_proba.append(clf.estimators_[j].predict_proba(X_test))
        test_pred = clf.predict(X_test)
    #-------------------------------------
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    for i in range(len(train_proba[0])):
        li = list(X_train[i])
        li.insert(0,'train')
        li.insert(0,(n_iter));
        li.append(y_train[i])
        li.append(train_pred[i])
        for j in range(N): # 여기서 20개 append
            li.append(train_proba[j][i][0]) # proba 0 추가 약분류기 j 번째에서 
            li.append(train_proba[j][i][1]) # proba 1추가
        frame = frame.append(pd.Series(li,index=idx_li),ignore_index=True)
    
    for i in range(len(test_proba[0])):
        li = list(X_test[i])
        li.insert(0,'test')
        li.insert(0,(n_iter));
        li.append(y_test[i])
        li.append(test_pred[i])
        for j in range(N): # 여기서 20개 append
            li.append(test_proba[j][i][0]) # proba 0 추가
            li.append(test_proba[j][i][1]) # proba 1추가
        frame = frame.append(pd.Series(li,index=idx_li),ignore_index=True)

    frame.to_csv("ResultRUSROS/inputdata_"+name+str(len(xlsx))+'_result.csv')
    n = n + 1

    result = ToExcel(train_acc, train_geoacc, train_auc,test_acc, test_geoacc, test_auc, Ths)
    result.to_csv("ResultRUSROS/result_"+name+str(len(xlsx))+".csv")
    w = pd.DataFrame(np.concatenate(w, axis=0))
    w.to_csv("ResultRUSROS/weight_"+name+str(len(xlsx))+".csv")

    #-------------------------- n 관련된 전처리 
    fold_n = []
    for i in range(n):
        fold_n.append(["fold_"+str(i)]*2)
    #--------------------------
    conmat = pd.DataFrame(np.concatenate(conmat,axis=1),columns = fold_n, index= ["train"]*2+["test"]*2)
    conmat.to_csv("ResultRUSROS/conmat_"+name+str(len(xlsx))+".csv")
    return clf



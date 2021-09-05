from Module.base import *
from Module.boost import *

import pandas as pd
import numpy as np
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _check_sample_weight
from sklearn.model_selection import train_test_split
# import 해야하는 모듈들
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import collections
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pyperclip
import math
from functools import reduce
import operator
warnings.filterwarnings('ignore')

# 전역변수 데이터프레임 설정
global conmaA
conmaA=pd.DataFrame()
global conmaB
conmaB=pd.DataFrame()
global conmaC
conmaC=pd.DataFrame()

# confusion matrix 담을 것 + excel 출력 대상임.
conma=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])

def roc_curve_plot(y_test , pred_proba_c1):
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)
    plt.plot(fprs , tprs, label='ROC')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()



def new_auc(y_test,pred):
    conma=confusion_matrix(y_test , pred) # y_test부분에 실제값, 2번째 파라미터로 인자값 
    TPR = conma[1,1]/(conma[1,0]+conma[1,1])
    FPR = conma[0,1]/(conma[0,0]+conma[0,1])
    return (1+TPR-FPR)/2

def Maketable(clf,X,y): # X = X_train, y = y_train
    table = pd.DataFrame(np.array(y)) # 그냥 y하면 데이터프레임이라서 기업 인덱스가 존재함
    for i, estimator in enumerate(clf.estimators_):
        ap = pd.DataFrame(estimator.predict(X), columns = ['0_'+str(i+1),'1_'+str(i+1)] ) # 이게 아마 넘파이 일거고
        table = pd.concat([table,ap],axis=1)
    return table 

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def makeexcel(clf,X_train,X_test,y_train,y_test):
    for i in range(len(clf.estimators_)):
        X = pd.DataFrame()
        y = pd.DataFrame()
        proba1 = clf.estimators_[i].predict(X_train)
        proba2 = clf.estimators_[i].predict(X_test)
        proba = pd.DataFrame(np.vstack((proba1,proba2)))
        X = pd.concat([X_train,X_test],ignore_index = True)
        y = pd.concat([y_train,y_test],ignore_index = True)
        indexname = ['train']*int((len(X)*(9/10)))+['test']*int(len(X)/10)
        result = pd.concat([X,y,proba],axis=1,ignore_index = True)
        result.index = indexname
        result.to_excel('estimators/first_fold_estimators'+str(i+1)+'.xlsx')
        
        
def geoacc(mat):
    value = (mat.iloc[0,0] / sum(mat.iloc[0,:])) * (mat.iloc[1,1] / sum(mat.iloc[1,:]))
    return math.sqrt(value)

def makeconma(conma,mat):
    conma.iloc[0,0],conma.iloc[0,1],conma.iloc[1,0],conma.iloc[1,1]=mat[0,0],mat[0,1],mat[1,0],mat[1,1]
    return conma

def acccondition(clf):
    clf.accfit = True
    clf.newfit = False
    clf.weight = False  

    
def makecolname(n_fold,n_for):
    colname_first = [["fold_{}".format(i),"fold_{}".format(i)] for i in range( n_fold*n_for )] # 여기 15는 n_estimators 갯수
    colname_first = list( reduce( operator.add, colname_first ) )
    return colname_first
    
    
    
def ToExcel(cv_acc,cv_auc,cv_geoacc,mat,colname_first,colname_second):
    
    """    accauc = np.array(cv_acc + cv_auc)
    geoaccs = np.array(cv_geoacc + cv_geoacc)
    
    mat = np.vstack((np.array(mat), accauc, geoaccs))
    mat = pd.DataFrame(np.array(mat),
                index = ['실제0','실제1','accauc','geoacc'],    
                columns = [colname_first, colname_second])"""
    mat = pd.DataFrame({"ACC":cv_acc,"GEOACC":cv_geoacc,"AUROC":cv_auc})
    mat = mat.T
    
    return mat

    
    
def oncefit(clf, xlsx, n_fold):
    skfold = StratifiedKFold(n_splits=n_fold)
    n_iter=0
    n_for = 3
    
    # 각 훈련에서 acc auc geoacc 담을 곳 
    global conmaA
    conmaA=pd.DataFrame()
    global conmaB
    conmaB=pd.DataFrame()
    global conmaC
    conmaC=pd.DataFrame()
    global conmaD
    conmaD=pd.DataFrame()
    cv_acc_log = []
    cv_auc_log = []
    cv_geoacc_log = []
              
    # 데이터 프레임 컬럼만드는 것 
    colname_first = makecolname(n_fold,n_for)
    for i in range(n_for):
        print("================================================",i,"번째================================================")
        xlsx.sample(frac=1,random_state=i).reset_index(drop=True)  
        X = xlsx.iloc[:,4:11]
        y = xlsx.iloc[:,11]

        for train_index, test_index  in skfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            n_iter += 1
            
            #학습 및 예측 
            clf.fit(X_train , y_train)                
            proba1 = clf.predict_proba(X_train)[:,1]
            Threshold = Find_Optimal_Cutoff(y_train, proba1)
            print(Threshold)
            proba2 = clf.predict_proba(X_test)[:,1]
            pred = proba2 >= Threshold
            print(pred)
            mat=confusion_matrix(y_test,pred)
            conma_temp = makeconma(conma,mat) # 이걸로 위에줄삭제
            
            # 평가지표 생성
            accuracy = np.round(accuracy_score(y_test,pred), 4)
            auc=roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            geoaccvalue = geoacc(conma_temp)
            
            # 평가지표담기
            cv_acc_log.append(accuracy)
            cv_auc_log.append(auc)
            cv_geoacc_log.append(geoaccvalue)
            conmaA=pd.concat([conmaD,conma_temp],axis=1)
            print()
           
            ## 임계점 추가한 디폴트핏 훈련 종료
    
    colname_second = conmaA.columns        
    
    # 이 아래 부분은 따로 만들지 말고 다 합쳐야 겠는데?  생각해보니깐 그러면 안댐 걍 따로 만들어야함 
    # Default fit 부분 
    resultdef = ToExcel(cv_acc_log,cv_auc_log,cv_geoacc_log,conmaA,colname_first,colname_second) 
    resultdef.to_excel('Result/logistic'+str(len(xlsx)) + '.xlsx')
    

def readexcel(roadname, datanumlist):
    Datalist = []
    for i,c in enumerate(datanumlist):
        Datalist.append(pd.read_excel(roadname+str(c)+'.xlsx'))
    return Datalist

def readcsv(roadname ,datanumlist):
    Datalist = []
    for i,c in enumerate(datanumlist):
        Datalist.append(pd.read_csv(roadname+str(c)+'.csv'))
    return Datalist

def valaccauc(datas): 
    for data in datas :
        reslen = int((data.shape[1]) / 2)
        accs = np.array(data.iloc[4,1:reslen+1])
        aucs = np.array(data.iloc[4,reslen+1:data.shape[1]])
        geoaccs = np.array(data.iloc[5,1:reslen+1])
        
        print( '-' * 30 )
        print(' acc_mean : %.2f \n acc_std : %.2f \n auc_mean : %.2f \n auc_std : %.2f \n geoacc_mean : %.2f \n geoacc_std : %.2f ' 
              % (np.mean(accs),np.std(accs), np.mean(aucs), np.std(aucs), np.mean(geoaccs), np.std(geoaccs)) )
        print( '-' * 30 )    





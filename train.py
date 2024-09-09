import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import  LogisticRegression
from  sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
heard_dara=pd.read_csv('heart.csv')
X=heard_dara.drop(columns='target',axis=1)
y=heard_dara['target']
x=np.asarray(X)
Y=np.asarray(y)
models=[LogisticRegression(max_iter=10000),SVC(),KNeighborsRegressor(),RandomForestClassifier(random_state=0)]
def compare():
    for model in models:
        cs_scoe=cross_val_score(model,X,Y,cv=5)
        mean=sum(cs_scoe)/len(cs_scoe)
        mean=mean*100
        mean=round(mean,2)
        print('accuracies,',cs_scoe)
        print('acc score of odel',mean)


model_paras={'log_reg_hyperparameters':
                 {'C':[1,5,10,20]
                  },
             'svc_hyperparameters':{
                 'kernel':['linear','poly','rbf','sigmoid'],
                 'C':[1,5,10,20]
             },
             'KNN_hyperparameters':{
                 'n_neighbours':[3,5,10]
             },
             'random_forest_classifier':{
                 'n_estimators':[10,20,50,100]
             }
             }

type(model_paras)

model_keys=list(model_paras.keys())


def model_selection(list_of_model,hyperparameters_dictionary):
    result=[]
    i=0
    for model in list_of_model:
        key=model_keys[i]
        params=hyperparameters_dictionary[key]
        i+=1
        print(model)
        print(params)

        classifier=GridSearchCV(model,params,cv=5)
        classifier.fit(X,Y)
        result.append({
            'model used':model,
            'highest score':classifier.best_score_,
        'best hyper':classifier.best_params_
        })
    result_dataframe=pd.DataFrame(result,columns=['model used','highest score','best hyper'])
    return result_dataframe

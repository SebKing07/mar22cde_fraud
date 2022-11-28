# Importation des librairies et instanciation de l’API:
from datetime import datetime
import uvicorn
from fastapi import FastAPI,Request, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Any, Dict,List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pickle
#from datetime import datetime 

## INSTANCIATION DE L’API
app = FastAPI(
  title="Fraud Detection API",
  description= "API pour la détection de la fraude à partir d’une modélisation ",
  version="0.1"
)
# Creating the data model for data validation
class Operation(BaseModel):
    user_id: str
    signup_time: datetime
    purchase_time: datetime
    purchase_value: float
    device_id: str
    source: str
    browser: str
    sex: str
    age: int
    ip_address: float

filename1 = "Model1.sav"
filename2 = "Model2.sav"
filename3 = "Model3.sav"

#Chargement des modèles

Model1 = pickle.load(open(filename1, 'rb'))
#Model2 = pickle.load(open(filename2, 'rb'))
#Model3 = pickle.load(open(filename3, 'rb'))


#Input Data Transformation

def prep(X):
    df=pd.DataFrame([X], columns=['user_id', 'signup_time', 'purchase_time', 'purchase_value',
                                   'device_id', 'source', 'browser', 'sex', 'age', 'ip_address'])
    LE = preprocessing.LabelEncoder()
    LE = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            LE=LabelEncoder()
            LE.fit(list(df[col].astype(str).values))
            df[col] = LE.transform(list(df[col].astype(str).values))  
            #df1=df.iloc[0,:].values
    return df

## API ENDPOINTS
## ----------------------------------------------------------------

# Test basique sur l'API
@app.get('/')
def index():
 '''
 This is a first docstring.
 '''
 return {'message': 'This is a Fraud Classification API!'}
# Tester le bon fonctionnement de l'API
@app.get('/ping')
def ping():
 '''
 This is a first docstring.
 '''
 return ('pong', 200)

# Definition du endpoint Model1
@app.get('/model1_predict')
async def model1_predict(user_id: str,
  signup_time: datetime,
  purchase_time: datetime,
  purchase_value: float,
  device_id: str,
  source: str,
  browser: str,
  sex: str,
  age: int,
  ip_address: float):
    
    #This is the first model prediction.

  # Définition du vecteur post prédiction
    operation_Listed_dict={'user_id':user_id,
                    "signup_time":signup_time,
                    "purchase_time":purchase_time,
                    "purchase_value":purchase_value,
                    "device_id":device_id,
                    "source":source,
                    "browser":browser,
                    "sex":sex,
                    "age":age,
                    "ip_address":ip_address
    }
    
    operation_Listed = pd.DataFrame(operation_Listed_dict, index=[0])
    #operation_Listed.reshape(1,10)
    operation_Listed['signup_time'] = operation_Listed['signup_time'].apply(lambda x: datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
    operation_Listed['purchase_time'] = operation_Listed['purchase_time'].apply(lambda x: datetime.strptime(str(x),"%Y-%m-%d %H:%M:%S"))
  
    #operation_Listed['signup_time'] = datetime.strptime(operation_Listed['signup_time'],"%Y-%m-%d %H:%M:%S").isoformat()
    #operation_Listed['purchase_time'] = datetime.strptime(operation_Listed['purchase_time'],"%Y-%m-%d %H:%M:%S").isoformat()
  
  
  # DataPreprocessing: Encodage
    #operation_Listed = prep(operation_Listed)
  #Prédiction
    pred1 = Model1.predict(operation_Listed)
    return {"prediction":pred1}
    #return operation_Listed
  # Definition du endpoint Model2
'''
@app.get('/model2_predict')
async def model2_predict(operation: Operation):



    # Définition du vecteur post prédiction
    operation_Listed=[operation.user_id,
                    operation.signup_time,
                    operation.purchase_time,
                    operation.purchase_value,
                    operation.device_id,
                    operation.source,
                    operation.browser,
                    operation.sex,
                    operation.age,
                    operation.ip_address
                     ]
  # DataPreprocessing: Encodage
    operation_Listed = prep (operation_Listed)
  #Prédiction
    pred2 = Model2.predict(operation_Listed)
    return pred2
    '''
''''
@app.get('/model3_predict')
async def model3_predict(operation: Operation):
    '''
    #This is the third model prediction.
'''
      # Définition du vecteur post prédiction
    operation_Listed=[operation.user_id,
                    operation.signup_time,
                    operation.purchase_time,
                    operation.purchase_value,
                    operation.device_id,
                    operation.source,
                    operation.browser,
                    operation.sex,
                    operation.age,
                    operation.ip_address
                     ]
 # DataPreprocessing: Encodage
    operation_Listed = prep (operation_Listed)
 #Prédiction
    pred3 = Model3.predict(operation_Listed)
    return pred3
'''
'''
@app.get('/perf_model1')
async def Model1_perf(operation: Operation):
    '''
    #display Model1 Performance.
'''
    return {"Roc Auc Score": Model1.roc_auc_score, "Accuracy": Model1.accuracy_score,"Precision_score": Model1.precision_score, "F1_score": Model1.f1_score}
'''
'''
@app.get('/perf_model2')
async def model2_perf(operation: Operation):
  '''
    #display Model2 Performance.
'''
    return {"Roc Auc Score": Model2.roc_auc_score, "Accuracy": Model2.accuracy_score,"Precision_score": Model2.precision_score, "F1_score": Model2.f1_score}
  '''
'''
@app.get('/perf_model3')
async def model3_perf(operation: Operation):
    '''
    #display Model3 Performance.
'''
    return {"Roc Auc Score": Model3.roc_auc_score, "Accuracy": Model3.accuracy_score,"Precision_score": Model3.precision_score, "F1_score": Model3.f1_score}
  '''
#uvicorn.run(app, port = 8000)
  
#? prediction = model.predict(df)
#? prediction_final=["Fraud" if (x > 0.5) else "Not Fraud" for x in prediction]
#?prediction_final
# Load the libraries
import codecs
import csv
from fastapi import FastAPI, HTTPException, File, UploadFile
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score # for calculating accuracy of model

import data_preprocess as dp

# Load the model
lsvm_binary_model = load(open('./models/lsvm_binary.pkl','rb'))
qsvm_binary_model = load(open('./models/qsvm_binary.pkl','rb'))
lda_binary_model = load(open('./models/lda_binary.pkl','rb'))
qda_binary_model = load(open('./models/qda_binary.pkl','rb'))
lsvm_multi_model = load(open('./models/lsvm_multi.pkl','rb'))
qsvm_multi_model = load(open('./models/qsvm_multi.pkl','rb'))
lda_multi_model = load(open('./models/lda_multi.pkl','rb'))
qda_multi_model = load(open('./models/qda_multi.pkl','rb'))

# dataset doesn't have column names, so we have to provide it
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty_level"]

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Intrusion Detection"}


# Define the route to the models classifier
@app.post("/binary-classification-using-lsvm")
def lsvm_binary(file: UploadFile = File(...)):

    cdf = pd.read_csv(file.file, header=None, names=col_names)
    file.file.close()

    print(cdf.head())

    df = dp.data_creation(cdf, 'binary')
    print(df)

    y_actual = df['intrusion']
    y_pred = lsvm_binary_model.predict(df.iloc[:,0:93].to_numpy())

    ac = accuracy_score(y_actual, y_pred)*100

    y_values = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            y_values.append('normal')
        else:
            y_values.append('abnormal')

    return {"accuracy_score": ac, "y_pred": y_values}


@app.post("/binary-classification-using-qsvm")
def qsvm_binary(file: UploadFile = File(...)):

    cdf = pd.read_csv(file.file, header=None, names=col_names)
    file.file.close()

    print(cdf.head())

    df = dp.data_creation(cdf, 'binary')
    print(df)

    y_actual = df['intrusion']
    y_pred = qsvm_binary_model.predict(df.iloc[:,0:93].to_numpy())

    ac = accuracy_score(y_actual, y_pred)*100

    y_values = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            y_values.append('normal')
        else:
            y_values.append('abnormal')

    return {"accuracy_score": ac, "y_pred": y_values}


@app.post("/binary-classification-using-lda")
def lda_binary(file: UploadFile = File(...)):

    cdf = pd.read_csv(file.file, header=None, names=col_names)
    file.file.close()

    print(cdf.head())

    df = dp.data_creation(cdf, 'binary')
    print(df)

    y_actual = df['intrusion']
    y_pred = lda_binary_model.predict(df.iloc[:,0:93].to_numpy())

    ac = accuracy_score(y_actual, y_pred)*100

    y_values = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            y_values.append('normal')
        else:
            y_values.append('abnormal')

    return {"accuracy_score": ac, "y_pred": y_values}


@app.post("/binary-classification-using-qda")
def qda_binary(file: UploadFile = File(...)):

    cdf = pd.read_csv(file.file, header=None, names=col_names)
    file.file.close()

    print(cdf.head())

    df = dp.data_creation(cdf, 'binary')
    print(df)

    y_actual = df['intrusion']
    y_pred = qda_binary_model.predict(df.iloc[:,0:93].to_numpy())

    ac = accuracy_score(y_actual, y_pred)*100

    y_values = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            y_values.append('normal')
        else:
            y_values.append('abnormal')

    return {"accuracy_score": ac, "y_pred": y_values}

@app.post("/multi-classification-using-lsvm")
def lsvm_multi(file: UploadFile = File(...)):

    cdf = pd.read_csv(file.file, header=None, names=col_names)
    file.file.close()

    print(cdf.head())

    df = dp.data_creation(cdf, 'multi')
    print(df)

    y_actual = df['intrusion']
    y_pred = lsvm_multi_model.predict(df.iloc[:,0:93].to_numpy())

    ac = accuracy_score(y_actual, y_pred)*100

    y_values = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            y_values.append('DoS')
        elif (y_pred[i] == 1):
            y_values.append('Probe')
        elif (y_pred[i] == 2):
            y_values.append('R2L')
        elif (y_pred[i] == 3):
            y_values.append('U2R')
        else:
            y_values.append('normal')

    return {"accuracy_score": ac, "y_pred": y_values}


@app.post("/multi-classification-using-qsvm")
def qsvm_multi(file: UploadFile = File(...)):

    cdf = pd.read_csv(file.file, header=None, names=col_names)
    file.file.close()

    print(cdf.head())

    df = dp.data_creation(cdf, 'multi')
    print(df)

    y_actual = df['intrusion']
    y_pred = qsvm_multi_model.predict(df.iloc[:,0:93].to_numpy())

    ac = accuracy_score(y_actual, y_pred)*100

    y_values = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            y_values.append('DoS')
        elif (y_pred[i] == 1):
            y_values.append('Probe')
        elif (y_pred[i] == 2):
            y_values.append('R2L')
        elif (y_pred[i] == 3):
            y_values.append('U2R')
        else:
            y_values.append('normal')

    return {"accuracy_score": ac, "y_pred": y_values}


@app.post("/multi-classification-using-lda")
def lda_multi(file: UploadFile = File(...)):

    cdf = pd.read_csv(file.file, header=None, names=col_names)
    file.file.close()

    print(cdf.head())

    df = dp.data_creation(cdf, 'multi')
    print(df)

    y_actual = df['intrusion']
    y_pred = lda_multi_model.predict(df.iloc[:,0:93].to_numpy())

    ac = accuracy_score(y_actual, y_pred)*100

    y_values = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            y_values.append('DoS')
        elif (y_pred[i] == 1):
            y_values.append('Probe')
        elif (y_pred[i] == 2):
            y_values.append('R2L')
        elif (y_pred[i] == 3):
            y_values.append('U2R')
        else:
            y_values.append('normal')

    return {"accuracy_score": ac, "y_pred": y_values}


@app.post("/multi-classification-using-qda")
def qda_multi(file: UploadFile = File(...)):

    cdf = pd.read_csv(file.file, header=None, names=col_names)
    file.file.close()

    print(cdf.head())

    df = dp.data_creation(cdf, 'multi')
    print(df)

    y_actual = df['intrusion']
    y_pred = qda_multi_model.predict(df.iloc[:,0:93].to_numpy())

    ac = accuracy_score(y_actual, y_pred)*100

    y_values = []
    for i in range(len(y_pred)):
        if (y_pred[i] == 0):
            y_values.append('DoS')
        elif (y_pred[i] == 1):
            y_values.append('Probe')
        elif (y_pred[i] == 2):
            y_values.append('R2L')
        elif (y_pred[i] == 3):
            y_values.append('U2R')
        else:
            y_values.append('normal')

    return {"accuracy_score": ac, "y_pred": y_values}


import numpy as np
import pandas as pd

import pickle # saving and loading trained model
from os import path

# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# importing library for plotting
import matplotlib.pyplot as plt

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

# changing attack labels to their respective attack class
def change_label(df):
  df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
  df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail',
       'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)
  df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
  df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)
  return df

def data_preprocess(df):
    df.drop(['difficulty_level'],axis=1,inplace=True)
    # print(df.shape)
    # print(df.describe())
    # print(df['label'].value_counts())
    df = change_label(df.copy())
    # print(df['label'].value_counts())
    return df

def data_normalization(df):
    # selecting numeric attributes columns from data
    numeric_col = df.select_dtypes(include='number').columns
    std_scaler = StandardScaler()
    for i in numeric_col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
    return df, numeric_col

def one_hot_encoding(df):
    train_data = pd.read_csv('./datasets/KDDTrain+.txt', header=None, names=col_names)

    # selecting categorical data attributes
    cat_col = ['protocol_type','service','flag']

    # creating a dataframe with only categorical attributes
    categorical = train_data[cat_col]

    # one-hot-encoding categorical attributes using pandas.get_dummies() function
    categorical = pd.get_dummies(categorical,columns=cat_col)

    return categorical

def binary_classification(df):
    # changing attack labels into two categories 'normal' and 'abnormal'
    bin_label = pd.DataFrame(df.label.map(lambda x:'normal' if x=='normal' else 'abnormal'))

    # creating a dataframe with binary labels (normal,abnormal)
    bin_data = df.copy()
    bin_data['label'] = bin_label

    # label encoding (0,1) binary labels (abnormal,normal)
    le1 = preprocessing.LabelEncoder()
    enc_label = bin_label.apply(le1.fit_transform)
    bin_data['intrusion'] = enc_label

    # dataset with binary labels and label encoded column
    bin_data.head()

    # one-hot-encoding attack label
    bin_data = pd.get_dummies(bin_data,columns=['label'],prefix="",prefix_sep="") 
    bin_data['label'] = bin_label
    
    return bin_data, le1

def multi_classification(df):
    # creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
    multi_data = df.copy()
    multi_label = pd.DataFrame(multi_data.label)

    # label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
    le2 = preprocessing.LabelEncoder()
    enc_label = multi_label.apply(le2.fit_transform)
    multi_data['intrusion'] = enc_label

    # one-hot-encoding attack label
    multi_data = pd.get_dummies(multi_data,columns=['label'],prefix="",prefix_sep="") 
    multi_data['label'] = multi_label

    return multi_data, le2

def binary_feature_ext(df, categorical, numeric_col):
    # creating a dataframe with only numeric attributes of binary class dataset and encoded label attribute 
    numeric_bin = df[numeric_col]
    numeric_bin['intrusion'] = df['intrusion']

    # finding the attributes which have more than 0.5 correlation with encoded attack label attribute 
    corr= numeric_bin.corr()
    corr_y = abs(corr['intrusion'])
    highest_corr = corr_y[corr_y >0.5]
    highest_corr.sort_values(ascending=True)

    # selecting attributes found by using pearson correlation coefficient
    numeric_bin = df[['count','srv_serror_rate','serror_rate','dst_host_serror_rate','dst_host_srv_serror_rate', 'logged_in','dst_host_same_srv_rate','dst_host_srv_count','same_srv_rate']]

    # joining the selected attribute with the one-hot-encoded categorical dataframe
    numeric_bin = numeric_bin.join(categorical)
    # then joining encoded, one-hot-encoded, and original attack label attribute
    df = numeric_bin.join(df[['intrusion','abnormal','normal','label']])

    return df

def multi_feature_ext(df, categorical, numeric_col):
    # creating a dataframe with only numeric attributes of multi class dataset and encoded label attribute 
    nummeric_df = df[numeric_col]
    nummeric_df['intrusion'] = df['intrusion']

    # finding the attributes which have more than 0.5 correlation with encoded attack label attribute 
    corr= nummeric_df.corr()
    corr_y = abs(corr['intrusion'])
    highest_corr = corr_y[corr_y >0.5]
    highest_corr.sort_values(ascending=True)

    # selecting attributes found by using pearson correlation coefficient
    nummeric_df = df[['count','srv_serror_rate','serror_rate','dst_host_serror_rate','dst_host_srv_serror_rate', 'logged_in','dst_host_same_srv_rate','dst_host_srv_count','same_srv_rate']]

    # joining the selected attribute with the one-hot-encoded categorical dataframe
    nummeric_df = nummeric_df.join(categorical)
    # then joining encoded, one-hot-encoded, and original attack label attribute
    df = nummeric_df.join(df[['intrusion','Dos','Probe','R2L','U2R','normal','label']])

    return df

def data_creation(df, type):
  print(df)

  df = data_preprocess(df.copy())
  print(df)

  df, col_nums = data_normalization(df.copy())

  df_categorical = one_hot_encoding(df.copy())
  print(df_categorical.head())

  if (type == 'binary'):
    df, le = binary_classification(df.copy())
    df = binary_feature_ext(df.copy(), df_categorical, col_nums)
  else:
    df, le = multi_classification(df.copy())
    df = multi_feature_ext(df.copy(), df_categorical, col_nums)

  return df
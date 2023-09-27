#!/usr/bin/env python
# coding: utf-8

# Summary:
# 
# The best score f1 - 0.69 - I was obtained using a model with 6 linear layers, batchnorms, and dropouts, on features with upsampling. Dropouts drastically increase the score from 0.2. Classic ML models (random forest, catboost) do not provide such a good result - the score was f1=0,6. But without upsampling, even classic models provide much worse result, because of the wrong prediction of small class.

# In[257]:


import pandas as pd
import numpy as np


# In[258]:


data = pd.read_csv('telco-customer-churn.csv')
data


# Посмотрим на данные. Столбец с номером телефона сразу можно удалить. Столбец с годом тоже (везде 2015 год). Потом нужно сгруппировать по id клиентов (какие-то столбцы просуммировать, какие-то - посчитать среднее). 

# In[259]:


data.corr()


# In[260]:


data.isna().any().sum()


# In[261]:


y = data['churn']
X = data.drop(['churn', 'callingnum', 'year', 'month', 'noadditionallines'], axis=1)


# In[262]:


X_num = X.select_dtypes(include='number')
X_num


# In[263]:


X_cat = X.select_dtypes(include='object')
X_cat


# In[264]:


X_cat_num = pd.get_dummies(X_cat)
X_cat_num


# In[265]:


X_all = pd.concat((X_num, X_cat_num), axis=1)
X_all['churn'] = y
X_all


# In[266]:


plt.figure(figsize=(15,8))
X_all.corr()['churn'].sort_values(ascending = False).plot(kind='bar')


# In[267]:


X_mean = X_all.groupby('customerid').mean()
X_sum = X_all.groupby('customerid').sum()
X_mean['totalcallduration'] = X_sum['totalcallduration']
y = X_mean['churn']
X_mean = X_mean.drop('churn', axis=1)
X_mean


# In[268]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_mean, y, test_size=0.3)


# In[269]:


X_train


# In[270]:


y_train


# In[271]:


X_test


# In[272]:


X_num.columns


# In[273]:


X_train_num = X_train[X_num.drop('customerid', axis=1).columns]
X_test_num = X_test[X_num.drop('customerid', axis=1).columns]
X_train_cat = X_train.drop(X_num.drop('customerid', axis=1).columns, axis=1)
X_test_cat = X_test.drop(X_num.drop('customerid', axis=1).columns, axis=1)
X_train_cat


# In[274]:


X_train_num


# In[275]:


from sklearn.feature_selection import SelectKBest, f_classif

X_train = pd.concat((X_train_num, X_train_cat), axis=1)
X_test = pd.concat((X_test_num, X_test_cat), axis=1)

slc = SelectKBest(k=30)
X_new = slc.fit_transform(X_train, y_train)
cols_idx = slc.get_support(indices=True)
X_test_new = X_test.iloc[:, cols_idx]
X_new_df = pd.DataFrame(data=X_new, index=X_train.index, columns=X_test_new.columns)
X_train = X_new_df
X_test = X_test_new
X_train


# In[276]:


from sklearn.preprocessing import StandardScaler

'''X_train_num = X_train.iloc[:,:7]
X_test_num = X_test.iloc[:,:7]
X_train_cat = X_train.iloc[:,7:]
X_test_cat = X_test.iloc[:,7:]'''

scaler = StandardScaler()
scaler.fit(X_train_num)
X_train_num_np = scaler.transform(X_train_num)
X_test_num_np = scaler.transform(X_test_num)


# In[277]:


X_train_cat


# In[278]:


X_test_num_sc = pd.DataFrame(data=X_test_num_np, index=X_test_num.index, columns=X_test_num.columns)
X_train_num_sc = pd.DataFrame(data=X_train_num_np, index=X_train_num.index, columns=X_train_num.columns)
X_train_num_sc


# In[279]:


X_test_num_sc


# In[280]:


X_train = pd.concat((X_train_num_sc, X_train_cat), axis=1)
X_test = pd.concat((X_test_num_sc, X_test_cat), axis=1)
X_test


# In[315]:


X_train['churn'] = y_train
X_train[X_train['churn'] == 0]


# In[316]:


X_test['churn'] = y_test
X_test[X_test['churn'] == 1]


# In[317]:


X_train_i = X_train.iloc[:, :-1]
y_train_i = X_train.iloc[:, -1]
X_test_i = X_test.iloc[:, :-1]
y_test_i = X_test.iloc[:, -1]


# In[318]:


X_train_i


# In[338]:


X_test_i


# In[283]:


from sklearn.utils import resample

X_majority = X_train[X_train['churn'] == 0]
X_minority = X_train[X_train['churn'] == 1]
 
# Upsample minority class
df_minority_upsampled = resample(X_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=6058,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([X_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.churn.value_counts()


# In[284]:


X_test.churn.value_counts()


# In[285]:


X_majority_test = X_test[X_test['churn'] == 0]
X_minority_test = X_test[X_test['churn'] == 1]

# Upsample minority class
df_minority_upsampled_test = resample(X_minority_test, 
                                 replace=True,     # sample with replacement
                                 n_samples=2606,    # to match majority class
                                 random_state=0) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled_test = pd.concat([X_majority_test, df_minority_upsampled_test])

# Display new class counts
df_upsampled_test.churn.value_counts()


# In[286]:


df_upsampled_test


# In[287]:


X_train = df_upsampled.iloc[:, :-1]
y_train = df_upsampled.iloc[:, -1]
data_train = X_train.copy()
data_train['churn'] = y_train
#X_train = X_train.drop('totalcallduration', axis=1)
#X_test = X_test.drop('totalcallduration', axis=1)
X_test = df_upsampled_test.iloc[:, :-1]
y_test = df_upsampled_test.iloc[:, -1]
data_test = X_test.copy()
data_test['churn'] = y_test
data_test


# In[288]:


plt.figure(figsize=(15,8))
data_train.corr()['churn'].sort_values(ascending = False).plot(kind='bar')


# In[289]:


import seaborn as sns


# In[290]:


X_test


# In[291]:


data_train = data_train.sample(frac = 1)
data_test = data_test.sample(frac = 1)
data_test


# In[292]:


X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:,-1]
X_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:,-1]


# In[293]:


X_test


# In[294]:


X_train


# In[295]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=2, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print(f1_score(y_train, y_pred), f1_score(y_test, y_pred_test))


# In[296]:


import xgboost as xgb

xgb_estimator = xgb.XGBClassifier(max_depth=2, reg_alpha=1)
xgb_estimator.fit(X_train, y_train)
y_pred = xgb_estimator.predict(X_train)
y_pred_test = xgb_estimator.predict(X_test)

print(f1_score(y_train, y_pred), f1_score(y_test, y_pred_test))


# In[297]:


y_pred_test.sum()/len(y_pred_test), y_test.sum()/len(y_test), y_train.sum()/len(y_train)


# In[298]:


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=10).fit(X_train, y_train)
y_pred = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

print(f1_score(y_train, y_pred), f1_score(y_test, y_pred_test))


# In[299]:


y_pred_test.sum()


# In[300]:


y_pred_test.sum()/len(y_pred_test), y_test.sum()/len(y_test)


# In[301]:


from sklearn.svm import SVC

svm = SVC(kernel='linear') 
svm.fit(X_train,y_train)
preds = svm.predict(X_test)
preds_train = svm.predict(X_train)
print(f1_score(y_train, preds_train), f1_score(y_test, preds))


# In[302]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=300)
neigh.fit(X_train, y_train)
y_pred1 = neigh.predict(X_test)
preds_train = neigh.predict(X_train)
print(f1_score(y_train, preds_train), f1_score(y_test, y_pred1))


# In[303]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred1))


# In[304]:


import torch.nn as nn
import torch
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Dataset, DataLoader
import copy
import torch.nn.functional as F


# In[305]:


n = X_train.shape[1]
n


# In[306]:


# Model with 4 linear layers

'''class Classify1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(n, n)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(n, n)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(n, n)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(n, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x'''


# In[332]:


# More complicated model 
class Classify1(nn.Module):
    def __init__(self):
            super(Classify1, self).__init__()
            self.hidden_size = [n, 100, 200, 100, n]
            self.dropout_value = [0.8, 0.8, 0.8, 0.8, 0.8]

            self.batch_norm1 = nn.BatchNorm1d(n)
            self.dense1 = nn.Linear(n, self.hidden_size[0])

            self.batch_norm2 = nn.BatchNorm1d(self.hidden_size[0])
            self.dropout2 = nn.Dropout(self.dropout_value[0])
            self.dense2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])

            self.batch_norm3 = nn.BatchNorm1d(self.hidden_size[1])
            self.dropout3 = nn.Dropout(self.dropout_value[1])
            self.dense3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])

            self.batch_norm4 = nn.BatchNorm1d(self.hidden_size[2])
            self.dropout4 = nn.Dropout(self.dropout_value[2])
            self.dense4 = nn.Linear(self.hidden_size[2], self.hidden_size[3])

            self.batch_norm5 = nn.BatchNorm1d(self.hidden_size[3])
            self.dropout5 = nn.Dropout(self.dropout_value[3])
            self.dense5 = nn.Linear(self.hidden_size[3], self.hidden_size[4])
            
            self.batch_norm6 = nn.BatchNorm1d(self.hidden_size[4])
            self.dropout6 = nn.Dropout(self.dropout_value[4])
            self.dense6 = nn.Linear(self.hidden_size[4], 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))
        
        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = F.leaky_relu(self.dense5(x))

        x = self.batch_norm6(x)
        x = self.dropout6(x)
        x = self.dense6(x)
        x = self.sigmoid(x)
        return x


# In[333]:


def model_train(model, X_train, y_train, X_val, y_val, n_epochs, batch_size, lr):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    batch_start = torch.arange(0, len(X_train), batch_size)

    best_f1 = 0
    best_weights = None
    total_losses = []
 
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # batches
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward
                y_pred = model(X_batch).T[0]
                #print('y_pred', y_pred)
                #print('y_batch', y_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                #print('loss', loss, 'epoch', epoch)
                bar.set_postfix(
                    loss=float(loss),
                )
        # evaluate loss at end of each epoch on the validation data and save this data to build diagram
        model.eval()
        y_pred_val = model(X_val).T[0]
        
        f1 = f1_score(y_val.detach().numpy(), y_pred_val.detach().numpy().round())
        print('epoch: ', epoch, 'f1: ', f1)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = copy.deepcopy(model.state_dict())
        total_losses.append(f1)
    # restore model and return best loss
    model.load_state_dict(best_weights)
    return best_f1, np.array(total_losses)


# In[334]:


X_train_t = torch.tensor(X_train_i.to_numpy()).float()
y_train_t = torch.tensor(y_train_i.to_numpy()).float()

# function for Folds generation
def Folds(model, X_train_t, y_train_t, n_epochs, batch_size, lr):

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    scores = []
    loss_arrays = []
    for i, (train, test) in enumerate(kfold.split(X_train_t, y_train_t)):
        # train and get validation loss for each batch
        losses = model_train(model, X_train_t[train], y_train_t[train], X_train_t[test], y_train_t[test], n_epochs, batch_size, lr)
        loss = losses[0]
        loss_array = losses[1]
        print("loss for iteration", loss)
        scores.append(loss)
        loss_arrays.append(loss_array)

    # get mean loss calculated with losses of 5 folds
    total_loss = np.mean(scores)
    loss_std = np.std(scores)
    print("Total loss: ", total_loss, '+-', loss_std)
    return np.array(loss_arrays)


# In[335]:


model1 = Classify1()
n_epochs=20
lr=0.01
batch_size=32
loss_arrays = Folds(model1, X_train_t, y_train_t, n_epochs=n_epochs, batch_size=batch_size, lr=lr)


# In[336]:


X_test_i


# The f1-score on a test data is 0,69.

# In[337]:


X_test_t = torch.tensor(X_test_i.to_numpy()).float()

model1.eval()
with torch.no_grad():
    y_pred_final = model1(X_test_t)
y_final = np.array(y_pred_final)
f1_score(y_test_i, y_final.round())


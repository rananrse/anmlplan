# MODEL
# importing necessary libraries for PY model file
import numpy as np
import pandas as pd
import openpyxl
import pickle
from pickle import load
from pickle import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score,mean_squared_error
#import warnings
#warnings.filterwarnings('ignore')
prefinal_dataset= pd.read_excel('prefinal_dataset3_for_board_analyse.xlsx')
#  convert CXO_date to numerical for feature engineering

prefinal_dataset['CXO_date'] = pd.to_datetime(prefinal_dataset['CXO_date'])

# Extract numerical features from the date
prefinal_dataset['CXO_year'] = prefinal_dataset['CXO_date'].dt.year
prefinal_dataset['CXO_month'] = prefinal_dataset['CXO_date'].dt.month
prefinal_dataset['CXO_day'] = prefinal_dataset['CXO_date'].dt.day
prefinal_dataset = prefinal_dataset[['board number', 'CXO_year', 'CXO_month', 'CXO_day','CXO_date', 'TF3', 'TF900', 'RP', 'TQ3SFF', 'TQ3SFDO','TQ3DF Fxd', 'TQ3DFDO','total_verticals', 'INFO', 'type_board','FAULT LEVEL', 'BUSBAR LOCATION', 'FORM OF SEPARATION', 'FIXED/DRAWOUT','SYSTEM',
                                  'number_of_bus', 'number_2_tier', 'EQP', 'FINAL TP','CAT number', 'iqr', 'average_cat_lead_time', 'no_of_days_MFG','Vendor name','CXO2DIN_diff','Billing Date','PRD_RED_MFG_Days']]
# data preprocessing and featuring
prefinal_dataset= prefinal_dataset.drop(['board number','CXO_date','CAT number','Vendor name','Billing Date'], axis=1)
prefinal_dataset = prefinal_dataset.fillna(0)
prefinal_dataset['INFO']=prefinal_dataset['INFO'].replace({'TF+TQ':'1', 'T-ERA':'2','CDO+':  '3'})
prefinal_dataset['type_board']=prefinal_dataset['type_board'].replace({'MCC':'1', 'PCC':'2','PMCC':  '3'})
prefinal_dataset['FAULT LEVEL']=prefinal_dataset['FAULT LEVEL'].replace(to_replace=['50KA FOR 1 SEC', '65KA FOR 1 SEC','80KA FOR 1 SEC'], value=['1', '2','3'], regex=True)
prefinal_dataset['BUSBAR LOCATION']=prefinal_dataset['BUSBAR LOCATION'].replace(to_replace=['TOP', 'BOTTOM'], value=['1', '2'], regex=True)
prefinal_dataset['FORM OF SEPARATION'] = prefinal_dataset['FORM OF SEPARATION'].replace(to_replace=['FORM3B', 'FORM4A','FORM4B','FORM4BTYPE6','FORM4BTYPE7'],value=['1','2','3','4','5'], regex=True)
prefinal_dataset['FIXED/DRAWOUT']=prefinal_dataset['FIXED/DRAWOUT'].replace(to_replace=['DRAWOUT', 'FIXED'], value=['1', '2'], regex=True)
prefinal_dataset['SYSTEM'] = prefinal_dataset['SYSTEM'].replace({'TP (3PH3W)': '1', 'TP+50%N (3PH4W)': '2', 'TP+100%N (3PH4W)': '3'})
# CONVERT ALL PREFINAL_DATASET STRING TO FLOAT

# Convert specified columns to numeric, coercing errors to NaN
for col in ['INFO', 'type_board', 'FAULT LEVEL', 'BUSBAR LOCATION', 'FORM OF SEPARATION', 'FIXED/DRAWOUT', 'SYSTEM','EQP']:
    prefinal_dataset[col] = pd.to_numeric(prefinal_dataset[col], errors='coerce')

# Fill NaN values resulting from conversion errors (if any)
prefinal_dataset = prefinal_dataset.fillna(0)
#split the dataset for test and train
X= prefinal_dataset.iloc[:,:-1]# all row and columns except last
y= prefinal_dataset.iloc[:,-1] # all row and column only from last
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# Random forest regressor model preparation for optimal model
# Define a range of ccp_alpha values to explore
ccp_alphas = np.linspace(0, 0.1, 10)  # Adjust the range as needed

# Train a random forest for each alpha value
clfs = []
for ccp_alpha in ccp_alphas:
    clf = RandomForestRegressor(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Evaluate each model on the training and test sets
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Find the best alpha value based on test scores
optimal_alpha = ccp_alphas[np.argmax(test_scores)]


# Train the final pruned model using the optimal alpha
optimal_rf_regressor = RandomForestRegressor(ccp_alpha=optimal_alpha)
optimal_rf_regressor.fit(X_train, y_train)

# Evaluate the pruned model
optimal_y_pred = optimal_rf_regressor.predict(X_test)
optimal_r2 = r2_score(y_test, optimal_y_pred)

optimal_y_pred_train = optimal_rf_regressor.predict(X_train)
train_accuracy_optimal_rf = r2_score(y_train, optimal_y_pred_train)


test_accuracy_optimal_rf = r2_score(y_test, optimal_y_pred)

pickle.dump(optimal_rf_regressor, open('optimal_rf_regressor.pkl','wb'))


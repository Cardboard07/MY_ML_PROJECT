import sys
sys.path.insert(0, r"C:\Users\hrish\Projects\FIRST_ML_PROJECT")

import pandas as pd
from src.logger import logger
from sklearn.model_selection import train_test_split
data=pd.read_csv(r'C:\Users\hrish\Downloads\wine+quality\winequality-red.csv',sep=";")
logger.info('data loaded')
data.head()
data.tail()
data.describe()
data.quality.unique()
logger.info(data.isnull().sum())
logger.info(data.info())
data['quality']=data['quality'].transform(lambda x: x -3)
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
logger.info('data processed')

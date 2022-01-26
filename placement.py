import numpy as np
import pandas as pd
df = pd.read_csv('students_placement.csv')
print(df.describe())
X = df.drop(columns=['placed'])
y = df['placed']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
#train the model
print(X_train.head())
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test,y_pred))
#save the model in pickle format
import pickle 
pickle.dump(rf,open('model1.pkl','wb'))
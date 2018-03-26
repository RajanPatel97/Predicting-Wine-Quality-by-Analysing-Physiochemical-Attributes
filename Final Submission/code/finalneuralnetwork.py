
# coding: utf-8

# In[1]:


# Import pandas 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# Read in white wine data 
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')


# In[2]:


# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)


# In[3]:


y = wines.quality
X = wines.drop(['quality', 'residual sugar', 'free sulfur dioxide', 'type'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=46, stratify=y)

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train_scaled = scaler.transform(X_train)

# Scale the test set
X_test_scaled = scaler.transform(X_test)


# In[4]:


import pandas as pd 
import numpy as np

from keras import layers, optimizers, regularizers
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential

from keras.utils import plot_model
#from kt_utils import *
import keras.backend as K

from sklearn import preprocessing, model_selection 


# In[5]:


from keras import regularizers
# create model
model = Sequential()
# layer 1
model.add(Dense(9, input_dim=9, activation='relu', kernel_initializer='normal'))
#layer 2
model.add(Dense(60, activation='relu', kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#layer 3
model.add(Dense(60, activation='relu',kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#layer 4
model.add(Dense(1, activation='linear'))
# Compile model
from keras import metrics
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer, loss='mean_squared_error', metrics=[metrics.mae, metrics.categorical_accuracy])


# In[10]:


model.fit(x = X_train_scaled, y = y_train, epochs = 100,verbose=1, batch_size = 150,validation_split=0.1)


# In[11]:


preds = model.evaluate(x = X_test_scaled, y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[12]:


y_pred = model.predict(X_test_scaled)
from sklearn import metrics   
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  


# In[13]:


fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[14]:


model.summary()


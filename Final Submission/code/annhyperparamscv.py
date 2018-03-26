
# coding: utf-8

# In[9]:


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


# In[10]:


# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)


# In[11]:


y = wines.quality
X = wines.drop(['quality', 'residual sugar', 'free sulfur dioxide', 'type'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=46, stratify=y)

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train_scaled = scaler.transform(X_train)

# Scale the test set
X_test_scaled = scaler.transform(X_test)


# In[12]:


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras import layers, optimizers, regularizers
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras import metrics

from keras.utils import plot_model
#from kt_utils import *
import keras.backend as K
from sklearn import preprocessing, model_selection 


# In[13]:


def create_model():
    # create model
    model = Sequential()
    # layer 1
    model.add(Dense(9, input_dim=9, activation='relu', kernel_initializer='normal') )
    #layer 2
    model.add(Dense(50, activation='relu', kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #layer 4
    model.add(Dense(1, activation='linear'))

    # Compile model
    model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size = [100,150,200,250,300]
epochs = [5,10,20,40,60,80,100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train_scaled, y_train)


# In[14]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (abs(mean), stdev, param))


# In[15]:


#Use scikit-learn to grid search the number of neurons
def create_model(neurons=1):
    # create model
    model = Sequential()
    # layer 1
    model.add(Dense(9, input_dim=9, activation='relu', kernel_initializer='normal'))
    #layer 2
    model.add(Dense(neurons, activation='relu', kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #layer 3
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=150, verbose=0)
# define the grid search parameters
neurons = [1, 5, 10, 20, 40, 60, 80, 100]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train_scaled, y_train)


# In[16]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[17]:


def create_model(hidden_layers=1):
    # create model
    model = Sequential()
    # layer 1
    model.add(Dense(9, input_dim=9, activation='relu',kernel_initializer='normal'))
    
    #hidden layers
    for i in range(hidden_layers):
        # Add one hidden layer
        model.add(Dense(60, activation='relu',  kernel_initializer='normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    
    #layer 4
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_model, batch_size=150, verbose=0, epochs=100)
# define the grid search parameters
hidden_layers = [1,2,3,4]
param_grid = dict(hidden_layers = hidden_layers)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train_scaled, y_train)


# In[18]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[20]:


# Use scikit-learn to grid search the activation function
def create_model(activation='relu'):
    # create model
    model = Sequential()
    # layer 1
    model.add(Dense(9, input_dim=9, activation='relu', kernel_initializer='normal'))
    #layer 2
    model.add(Dense(60, activation=activation,kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #layer 3
    model.add(Dense(60, activation=activation,kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    #layer 4
    model.add(Dense(1, activation='linear'))
    
    # Compile model
    model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=150, verbose=0)
# define the grid search parameters
activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train_scaled, y_train)


# In[21]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[22]:


from keras.optimizers import Adam
def create_model(learn_rate=0.01):
    # create model
    model = Sequential()
    # layer 1
    model.add(Dense(9, input_dim=9, activation='relu', kernel_initializer='normal'))
    #layer 2
    model.add(Dense(60, activation='relu', kernel_initializer='normal'))
    #layer 3
    model.add(Dense(60, activation='relu',kernel_initializer='normal'))
    #layer 4
    model.add(Dense(1, activation='linear'))
    optimizer = Adam(lr=learn_rate)
    # Compile model
    model.compile(optimizer = optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model

import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=150, verbose=0)
# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learn_rate=learn_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train_scaled, y_train)


# In[23]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


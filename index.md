# Concepts and Applications of Multiple Boosting and LightGBM

### LightGBM
LightGBM is a fast, powerful, high-performance gradient boosting framework based on decision tree algorithm. It is used for ranking, classification, and many other machine learning tasks.

Since it is based on decision tree algorithms, it splits the tree leaf wise with the best fit (the tree grows vertically), whereas other boosting algorithms split the tree depth wise or level wise (their trees grow horizontally). Therefore, when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. 

Level-wise tree growth in XGBOOST:

<img width="724" alt="image" src="https://user-images.githubusercontent.com/98488324/156966728-56bb89bf-cadd-4e80-9b1a-d9a00ca4623e.png">

Leaf-wise tree growth in LightGBM:

<img width="796" alt="image" src="https://user-images.githubusercontent.com/98488324/156966754-a40a439b-7364-4cc8-85da-7fe520dfd817.png">

LightGBM is called “Light” because of its computation power and giving results faster. It takes less memory to run and is able to deal with large amounts of data. It is not for a small volume of datasets as it can easily overfit small data due to its sensitivity. It can be used for large volumes of data having more than 10,000+ rows, especially when one needs to achieve a high accuracy of results. It is the most widely used algorithm in Hackathons because the motive of the algorithm is to get good accuracy of results and also brace GPU leaning.


## Applications with Real Data
Concrete Compressive Strength dataset (output variable (y) is the concrete_compressive_strength): 

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("drive/MyDrive/DATA410_AdvML/concrete_data.csv")
```
<img width="1035" alt="image" src="https://user-images.githubusercontent.com/98488324/156958221-05f272ba-d0c2-4039-b604-084cb57311e7.png">


### Multiple Boosting
Import libraries and create functions:

```python
# import libraries
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import xgboost as xgb

# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
  
# defining the kernel local regression model
def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
  
# defining the kernel boosted lowess regression model
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  #model = DecisionTreeRegressor(max_depth=2, random_state=123)
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  #model = model_xgb
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output
  
def booster(X, y, xnew, kern, tau, model_boosting, nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new
```

#### Apply concrete data:

```python
X = df[['cement', 'water']].values
y = df['concrete_compressive_strength'].values

model_boosting = RandomForestRegressor(n_estimators=100,max_depth=3)
scale = StandardScaler()
xscaled = scale.fit_transform(X)

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.25, random_state=123)

# we want more nested cross-validations
mse_lwr = []
mse_blwr = []
mse_rf = []
# mse_nn = []
mse_xgb = []
# mse_NW = []

for i in range(5):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)

    yhat_lwr = lw_reg(xtrain,ytrain,xtest,Epanechnikov,tau=0.9,intercept=True)
    #yhat_blwr = boosted_lwr(xtrain,ytrain,xtest,Epanechnikov,tau=0.9,intercept=True)
    yhat_blwr = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
    model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
    model_rf.fit(xtrain,ytrain)
    yhat_rf = model_rf.predict(xtest)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    # model_nn.fit(xtrain,ytrain,validation_split=0.2, epochs=500, batch_size=10, verbose=0, callbacks=[es])
    # yhat_nn = model_nn.predict(xtest)
    # here is the application of the N-W regressor
    # model_KernReg = KernelReg(endog=dat_train[:,-1],exog=dat_train[:,:-1],var_type='ccc',ckertype='gaussian')
    # yhat_sm, yhat_std = model_KernReg.fit(dat_test[:,:-1])

    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_rf.append(mse(ytest,yhat_rf))
    mse_xgb.append(mse(ytest,yhat_xgb))
    # mse_nn.append(mse(ytest,yhat_nn))
    # mse_NW.append(mse(ytest,yhat_sm))
print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Boosted LWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for RF is : '+str(np.mean(mse_rf)))
# print('The Cross-validated Mean Squared Error for NN is : '+str(np.mean(mse_nn)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
# print('The Cross-validated Mean Squared Error for Nadarya-Watson Regressor is : '+str(np.mean(mse_NW)))
```

#### Final results: 

The Cross-validated Mean Squared Error for LWR is : 175.482983119759       
The Cross-validated Mean Squared Error for Boosted LWR is : 159.30179064112733       
The Cross-validated Mean Squared Error for RF is : 170.1605321899341       
The Cross-validated Mean Squared Error for XGB is : 172.00290255220014      

Since we aim to minimize the crossvalidated mean square error (MSE) for the better results, I conclude that Boosted Lowess achieved significantly better result than other regressions including Lowess, Random Forest, and Extreme Gradient Boosting (XGBoost). 
       
1. Create your own multiple boosting algortihm and apply it to combinations of different regressors (for example you can boost regressor 1 with regressor 2 a couple of times) on the "Concrete Compressive Strength" dataset.  Show what was the combination that achieved the best cross-validated results.
2. (Research) Read about the LightGBM algorithm and include a write-up that explains the method in your own words. Apply the method to the same data set you worked on for part 1. 


### LightGBM

```python

```

#### Final results: 
    



## References

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Dwivedi, R. (Jun 26, 2020). What is LightGBM Algorithm, How to use it? Analytics Steps [https://www.analyticssteps.com/blogs/what-light-gbm-algorithm-how-use-it].

Bachman, E. (June 12, 2017). Which algorithm takes the crown: Light GBM vs XGBOOST? Analytics Vidhya [https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/].

##### Jekyll Themes
Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/r-fukutoku/Project2/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

##### Support or Contact
Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

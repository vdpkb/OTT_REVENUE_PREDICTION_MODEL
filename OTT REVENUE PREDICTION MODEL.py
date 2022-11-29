#!/usr/bin/env python
# coding: utf-8

# # OTT REVENUE PREDICTION MODEL

# A model to predict the revenue in million dollars based on the number of scubscribers. 
# 
# Data set:
# 
# Independant variable X: Subscribers/Year/Content Spend/Profit  
# Dependant variable Y: Overall revenue generated in dollars

# In[1]:


#import required libraries
import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#List all files used
for dirname, _, filenames in os.walk(r"C:\Users\vdp10002\Desktop\MainProject_OTTRevenuePredictionModel"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


#import datset
df_profit = pd.read_csv(r"C:\Users\vdp10002\OneDrive - Advanced Micro Devices Inc\IISC_project\MainProject_OTTRevenuePredictionModel\Netflix\Profit.csv")
df_subscribers = pd.read_csv(r"C:\Users\vdp10002\Desktop\MainProject_OTTRevenuePredictionModel\Netflix\NumSubscribers.csv")
df_revenue = pd.read_csv(r"C:\Users\vdp10002\Desktop\MainProject_OTTRevenuePredictionModel\Netflix\Revenue.csv")
df_ContentSpend=pd.read_csv(r"C:\Users\vdp10002\Desktop\MainProject_OTTRevenuePredictionModel\Netflix\ContentSpend.csv")
#initial exploration of data
df_profit


# In[4]:


df_subscribers


# In[5]:


df_revenue


# In[6]:


df_ContentSpend


# # VISUALIZE DATSET

# In[7]:


#understanding the trend of Revenue over the years, profit and content spend
x1 = df_profit['Year'].values
y1 = df_profit['Profit'].values

x2=df_subscribers['Year'].values
y2=df_subscribers['Subscribers'].values

x3=df_revenue['Year'].values
y3=df_revenue['Revenue'].values

x4=df_ContentSpend['Year'].values
y4=df_ContentSpend['Content_spend'].values

plt.rcParams["figure.figsize"] = (15,7)

plt.plot(x1, y1, 'red', label='Profit',  marker='o', linestyle='-', linewidth='1')
#plt.plot(x2, y2, 'blue', label='Subscribers')
plt.plot(x3, y3, 'green', label='Revenue',  marker='o', linestyle='-', linewidth='1')
plt.plot(x4, y4, 'black', label='Content Spend',  marker='o', linestyle='-', linewidth='1')

plt.grid()
plt.xlabel('YEARS', fontweight="bold")
plt.ylabel('in Million Dollars', fontweight="bold")
plt.title('Revenue, Profit and Content Spend over the years', fontweight="bold")
plt.legend()
plt.show()


# In[8]:


#understanding the trend of Revenue wrt # of Subscribers
x5=df_subscribers['Year'].values
y5=df_subscribers['Subscribers'].values

plt.rcParams["figure.figsize"] = (15,7)

plt.plot(x5, y5, 'black', label='Subscribers', marker='o', linestyle='-', linewidth='1')

plt.grid()
plt.xlabel('YEARS', fontweight="bold")
plt.ylabel('#of Subscribers', fontweight="bold")
plt.title('Subscribers over the years', fontweight="bold")
plt.legend()
plt.show()


# In[9]:


df_new = pd.merge(pd.merge(df_profit,df_revenue,on='Year'),df_subscribers,on='Year', how='right')
df_new1 = pd.merge(df_new, df_ContentSpend, on='Year', how='outer')


# In[10]:


# developing a histogram using DISPLOT
sns.displot(data   = df_new1,
            x      = 'Revenue',
            height = 5,
            aspect = 2)


plt.show()


# In[11]:


sns.lmplot(x='Year', y='Revenue', data=df_new1)


# In[12]:


sns.lmplot(x='Subscribers', y='Revenue', data=df_new1)


# In[13]:


sns.lmplot(x='Content_spend', y='Revenue', data=df_new1)


# In[14]:


sns.lmplot(x='Profit', y='Revenue', data=df_new1)


# # MISSING VALUE ANALYSIS AND IMPUTATION

# In[15]:


df_new1.isnull()


# In[16]:


df_new1


# In[17]:


## Replace all NaN values with 0
#df_new2= df_new1.fillna(0)
#df_new2
df_new2=df_new1
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
print(imputer)
df_new2.Profit = imputer.fit_transform(df_new2['Profit'].values.reshape(-1,1))
df_new2.Revenue = imputer.fit_transform(df_new2['Revenue'].values.reshape(-1,1))
df_new2.Subscribers = imputer.fit_transform(df_new2['Subscribers'].values.reshape(-1,1))
df_new2.Content_spend = imputer.fit_transform(df_new2['Content_spend'].values.reshape(-1,1))
df_new2


# # Understand the relationship between variables

# In[18]:


sns.pairplot(df_new2)


# # OLS results interpretation

# In[19]:


from sklearn import linear_model
import statsmodels.api as sm

x = df_new2[['Year','Subscribers','Profit','Content_spend']]
y = df_new2['Revenue']
 
# adding a constant as an intercept is not included by default and has to be added manually 
x = sm.add_constant(x) 
 
model = sm.OLS(y, x).fit()
 
print_model = model.summary()
print(print_model)


# # Linear Regression Model

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import r2_score


# In[21]:


#split features: recognising dependent and independent variables
y=df_new2[['Revenue']]
print(y)
x=df_new2.drop(['Revenue'], axis=1)
print(x)


# In[22]:


#preparing training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
x_train.shape
print(x_train)


# In[23]:


x_test


# In[24]:


y_test


# In[25]:


y_train


# In[26]:


get_ipython().run_cell_magic('time', '', '#instantiating linear regression model and fitting it to the training data\nLR = LinearRegression()\nLR.fit(x_train,y_train)\n')


# In[27]:


print('Intercept (c): ', LR.intercept_)
print('Coefficient (m): ', LR.coef_)
#scoring the model based on training and testing data
LR_test_score=LR.score(x_test, y_test)
LR_train_score=LR.score(x_train, y_train)
print('LR Testing Score: ', LR_test_score)
print('LR Trainig Score: ', LR_train_score)


# In[28]:


#predicting on test data
y_predict = LR.predict(x_test)
print(y_predict)


# In[29]:


#evaluating Linear Regression model
MSE_LR=mean_squared_error(y_test,y_predict)
RMSE_LR=np.sqrt(mean_squared_error(y_test,y_predict))
print("LR mean_sqrd_error is==", MSE_LR)
print("LR root_mean_squared error of is==",RMSE_LR)


# In[30]:


#evaluating feature YEAR
x_yr=df_new2[['Year']]
y_yr=df_new2[['Revenue']]
x_yr_train, x_yr_test, y_yr_train, y_yr_test = train_test_split(x_yr, y_yr, test_size=0.25)
print('Shape of X_Year_TrainingData:', x_yr_train.shape)
LR_yr = LinearRegression()
LR_yr.fit(x_yr_train,y_yr_train)
y_yr_predict = LR_yr.predict(x_yr_test)
print(y_yr_predict)
print('Test Score:',LR_yr.score(x_yr_test, y_yr_test))
print('Train Score:', LR_yr.score(x_yr_train, y_yr_train))
print('Linear Model Coefficient (m): ', LR_yr.coef_)
print('Linear Model Coefficient (b): ', LR_yr.intercept_)
MSE_LR_yr=mean_squared_error(y_yr_test,y_yr_predict)
print("mean_sqrd_error is==", MSE_LR_yr)


# In[31]:


#evaluating feature CONTENT SPEND
x_cont=df_new2[['Content_spend']]
y_cont=df_new2[['Revenue']]
#print(x_sub)
#print(y_sub)
x_cont_train, x_cont_test, y_cont_train, y_cont_test = train_test_split(x_cont, y_cont, test_size=0.25)
#print('Shape of X_Subscribers_TrainingData:', x_sub_train.shape)
#print(x_sub_train)
LR_cont = LinearRegression()
LR_cont.fit(x_cont_train,y_cont_train)
y_cont_predict = LR_cont.predict(x_cont_test)
print(y_cont_predict)
print('Test Score:',LR_cont.score(x_cont_test, y_cont_test))
print('Train Score:', LR_cont.score(x_cont_train, y_cont_train))
print('Linear Model Coefficient (m): ', LR_cont.coef_)
print('Linear Model Coefficient (b): ', LR_cont.intercept_)
MSE_LR_cont=mean_squared_error(y_cont_test,y_cont_predict)
print("mean_sqrd_error is==", MSE_LR_cont)


# In[32]:


#evaluating feature PROFIT
x_prof=df_new2[['Profit']]
y_prof=df_new2[['Revenue']]
#print(x_sub)
#print(y_sub)
x_prof_train, x_prof_test, y_prof_train, y_prof_test = train_test_split(x_prof, y_prof, test_size=0.25)
#print('Shape of X_Subscribers_TrainingData:', x_sub_train.shape)
#print(x_sub_train)
LR_prof = LinearRegression()
LR_prof.fit(x_prof_train,y_prof_train)
y_prof_predict = LR_prof.predict(x_prof_test)
print(y_prof_predict)
print('Test Score:',LR_prof.score(x_prof_test, y_prof_test))
print('Train Score:', LR_prof.score(x_prof_train, y_prof_train))
print('Linear Model Coefficient (m): ', LR_prof.coef_)
print('Linear Model Coefficient (b): ', LR_prof.intercept_)
MSE_LR_prof=mean_squared_error(y_prof_test,y_prof_predict)
print("mean_sqrd_error is==", MSE_LR_prof)


# In[33]:


#evaluating feature SUBSCRIBER
x_sub=df_new2[['Subscribers']]
y_sub=df_new2[['Revenue']]
#print(x_sub)
#print(y_sub)
x_sub_train, x_sub_test, y_sub_train, y_sub_test = train_test_split(x_sub, y_sub, test_size=0.25)
#print('Shape of X_Subscribers_TrainingData:', x_sub_train.shape)
#print(x_sub_train)
LR_sub = LinearRegression()
LR_sub.fit(x_sub_train,y_sub_train)
y_sub_predict = LR_sub.predict(x_sub_test)
print(y_sub_predict)
print('Test Score:',LR_sub.score(x_sub_test, y_sub_test))
print('Train Score:', LR_sub.score(x_sub_train, y_sub_train))
print('Linear Model Coefficient (m): ', LR_sub.coef_)
print('Linear Model Coefficient (b): ', LR_sub.intercept_)
MSE_LR_sub=mean_squared_error(y_sub_test,y_sub_predict)
print("mean_sqrd_error is==", MSE_LR_sub)


# In[34]:


plt.bar(["Year", "Subscribers","Profit","Content_spend"],[MSE_LR_yr,MSE_LR_sub,MSE_LR_prof,MSE_LR_cont])
plt.title("Feature Selection", fontweight='bold')
plt.ylabel("MSE of features")
plt.xlabel("Features")
plt.show()


# In[35]:


# predict revenue with #ofSubscribers
y_predict_1 = LR_sub.predict([[200]])
y_predict_1


# In[36]:


# predict revenue for a particular year
y_predict_2 = LR_yr.predict([[2030]])
print(y_predict_2)


# # Random Forest Regression Model

# In[37]:


from sklearn.ensemble import RandomForestRegressor


# In[38]:


get_ipython().run_cell_magic('time', '', '#instantiating Random Forest regression model and fitting it to the training data\nRF = RandomForestRegressor(n_jobs=-1) \nRF.fit(x_sub_train, y_sub_train)\n')


# In[39]:


# Calculate R2
RF_train_score=RF.score(x_sub_train, y_sub_train)
RF_test_score=RF.score(x_sub_test, y_sub_test)
print('RF Testing Score:',RF_test_score)
print('RF Taining Score:',RF_train_score)


# In[40]:


#predicting on test data
y_predict_RF = RF.predict(x_sub_test)
print(y_predict_RF)


# In[41]:


#evaluating Random Forest regression model
score_RF=RF.score(y_sub_test,y_predict_RF)
MSE_RF=mean_squared_error(y_sub_test,y_predict_RF)
RMSE_RF=np.sqrt(mean_squared_error(y_sub_test,y_predict_RF))
print("RF r2 socre is ",score_RF)
print("RF mean_sqrd_error is==", MSE_RF)
print("RF root_mean_squared error of is==",RMSE_RF)


# # KNN Regression Model

# In[42]:


from sklearn.neighbors import KNeighborsRegressor 


# In[43]:


get_ipython().run_cell_magic('time', '', '#instantiating KNN regression model and fitting it to the training data\nKNN = KNeighborsRegressor(n_neighbors=2)\nKNN.fit(x_sub_train,y_sub_train)\n')


# In[44]:


KNN_test_score=KNN.score(x_sub_test, y_sub_test)
KNN_train_score=KNN.score(x_sub_train, y_sub_train)
print('KNN Testing Score:',KNN_test_score)
print('KNN Taining Score:',KNN_train_score)


# In[45]:


y_predict_knn = KNN.predict(x_sub_test)
print(y_predict_knn)


# In[47]:


score_KNN=KNN.score(y_sub_test,y_predict_RF)
MSE_KNN=mean_squared_error(y_sub_test,y_predict_knn)
RMSE_KNN=np.sqrt(mean_squared_error(y_sub_test,y_predict_knn))
print("r2 socre is ",score_KNN)
print("MSE is ",MSE_KNN)
print("RMSE is ",RMSE_KNN)


# # Decision Tree Regression Model

# In[48]:


from sklearn.tree import DecisionTreeRegressor 


# In[49]:


get_ipython().run_cell_magic('time', '', '#instantiating Decision Tree regression model and fitting it to the training data\nDT = DecisionTreeRegressor(random_state = 0)\nDT.fit(x_sub_train,y_sub_train)\n')


# In[50]:


DT_test_score=DT.score(x_sub_test, y_sub_test)
DT_train_score=DT.score(x_sub_train, y_sub_train)
print('DT Testing Score:',DT_test_score)
print('DT Taining Score:',DT_train_score)


# In[51]:


y_predict_DT = DT.predict(x_sub_test)
print(y_predict_DT)


# In[52]:


score_DT=DT.score(y_sub_test,y_predict_knn)
MSE_DT=mean_squared_error(y_sub_test,y_predict_DT)
RMSE_DT=np.sqrt(mean_squared_error(y_sub_test,y_predict_DT))
print("r2 socre is ",score_DT)
print("MSE is ",MSE_DT)
print("RMSE is ",RMSE_DT)


# # Comparing Models

# In[53]:


plt.bar(["Linear", "DecisionTree","RandomForest","KNN"],[MSE_LR,MSE_DT,MSE_RF,MSE_KNN], color=['blue', 'red', 'yellow', 'green'])
plt.title("Error comparision of ML models", fontweight="bold")
plt.ylabel("MSE", fontweight="bold")
plt.xlabel("Regression Models", fontweight="bold")
plt.show()


# In[ ]:





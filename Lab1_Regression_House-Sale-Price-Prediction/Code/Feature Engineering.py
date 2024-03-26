# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:11:22 2023

@author: User
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn import preprocessing
import xlwings as xw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

###########################################################################
##                         Initial Settings                              ##
###########################################################################
data_mode = 'test'
norm_mode = 'z-score'
name = 'test_data_norm_v2.csv'
train_data = 'D:/Master/2023ML/Lab1/train-v3.csv'
valid_data = 'D:/Master/2023ML/Lab1/valid-v3.csv'
test_data = 'D:/Master/2023ML/Lab1/test-v3.csv'
save = 'D:/Master/2023ML/Lab1/'

###########################################################################
##                         Calculation                                   ##
###########################################################################
def sale_long(yr, mth, day):
    return ( (2023-yr)*365 + (12-mth)*30 + (31-day))

def built_long(yr):
    return ( (2023-yr))

###########################################################################
##                           Sketch Data                                 ##
###########################################################################

feature_sel = ['sale_yr', 'sale_month', 'sale_day', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

df_train = pd.read_csv(train_data, usecols=feature_sel)
df_valid = pd.read_csv(valid_data, usecols=feature_sel)
df_test = pd.read_csv(test_data, usecols=feature_sel)

if data_mode == 'train':
    df = df_train
    df_id_price = pd.read_csv(train_data, usecols=['id', 'price'])
elif data_mode == 'valid':
    df = df_valid
    df_id_price = pd.read_csv(valid_data, usecols=['id', 'price'])
elif data_mode == 'test':
    df = df_test
    df_id_price = pd.read_csv(test_data, usecols=['id'])

length = len (df)
print(df)


###########################################################################
##                           Add new features                            ##
###########################################################################
time_feature = ['sale_yr', 'sale_month', 'sale_day', 'yr_built', 'yr_renovated']
layout_feature = ['bedrooms', 'bathrooms', 'floors', 'waterfront' ]
area_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
address_feature = ['zipcode', 'lat', 'long']
condition_feature = ['condition', 'grade']
#-------------------------------------------------------------------------------
kmeans_t = KMeans(n_clusters=3)
kmeans_t.fit(df[time_feature])
df['cust_time_type'] = kmeans_t.predict(df[time_feature])
#-------------------------------------------------------------------------------
kmeans_l = KMeans(n_clusters=3)
kmeans_l.fit(df[layout_feature])
df['cust_layout_type'] = kmeans_l.predict(df[layout_feature])
#-------------------------------------------------------------------------------
kmeans_ar = KMeans(n_clusters=3)
kmeans_ar.fit(df[area_feature])
df['cust_area_type'] = kmeans_ar.predict(df[area_feature])
#-------------------------------------------------------------------------------
kmeans_ad = KMeans(n_clusters=3)
kmeans_ad.fit(df[address_feature])
df['cust_address_type'] = kmeans_ad.predict(df[address_feature])
#-------------------------------------------------------------------------------
kmeans_c = KMeans(n_clusters=3)
kmeans_c.fit(df[condition_feature])
df['cust_condition_type'] = kmeans_c.predict(df[condition_feature])
#-------------------------------------------------------------------------------
df['sale_long']=df.apply(lambda r: sale_long(r['sale_yr'], r['sale_month'], r['sale_day']), axis=1)
df['more_than_1_floor']=df.floors.apply(lambda x:1 if x>1 else 0)
df['grade_more_than_8']=df.grade.apply(lambda x:1 if x>8 else 0)
df['basement_or_not']=df.sqft_basement.apply(lambda x:1 if x>0 else 0)
df['renovate_or_not']=df.yr_renovated.apply(lambda x:1 if x>0 else 0)
df['built_long']=df.apply(lambda r: built_long(r['yr_built']), axis=1)

'''
numeric_feature = ['bedrooms', 'bathrooms', 'sqft_living',
               'sqft_lot', 'floors', 'sqft_above',
               'sqft_basement', 'sqft_living15',
               'sqft_lot15', 'sale_long', 'built_long']
nonnumeric_feature = all_feature = ['sale_yr', 'sale_month', 'sale_day', 'waterfront', 'view', 'condition', 'grade',
                'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
               'cust_time_type', 'cust_layout_type', 'cust_area_type', 'cust_address_type', 'cust_condition_type', 'more_than_1_floor', 'grade_more_than_8', 'basement_or_not',
               'renovate_or_not']

print(df)

df_out = pd.concat([df], axis=1)
print(df_out)
out_f = pd.DataFrame(df_out)
save_to_excel_2 = pd.concat([out_f]) #-------改
save_to_excel_2.to_csv(save+name)




### Visualization
#sns.pairplot(data=df, diag_kind='kde')
plt.figure(figsize=(30,30))
sns.heatmap(df[['price','sale_yr', 'sale_month', 'sale_day', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15','cust_time_type', 'cust_layout_type', 'cust_area_type', 'cust_address_type', 'cust_condition_type', 'sale_long', 'more_than_1_floor', 'grade_more_than_8', 'basement_or_not', 'renovate_or_not', 'built_long']].corr(), cmap='Blues', annot=True)
plt.show()


'''

###########################################################################
##                The maximum absolute scaling  [-1,1]                   ##
###########################################################################
# copy the data
df_max_scaled = df.copy()
  
# apply normalization techniques
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()
    
###########################################################################
##                 The min-max feature scaling  [0,1]                    ##
###########################################################################    
# copy the data
df_min_max_scaled = df.copy()
  
# apply normalization techniques
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())        


###########################################################################
##                       The z-score method                              ##
########################################################################### 
# copy the data
df_z_scaled = df.copy()
  
# apply normalization techniques
for columns in df_z_scaled.columns:
    df_z_scaled[columns] = (df_z_scaled[columns] -
                           df_z_scaled[columns].mean()) / df_z_scaled[columns].std()    
  

###########################################################################
##                            Save Results                               ##
###########################################################################

if norm_mode == 'max_abs':
    df_out = df_max_scaled
elif norm_mode == 'min_max':
    df_out = df_min_max_scaled
elif norm_mode == 'z-score':
    df_out = df_z_scaled

df_out = pd.concat([df_id_price, df_out], axis=1)
print(df_out)
out_f = pd.DataFrame(df_out)
save_to_excel_2 = pd.concat([out_f]) #-------改
save_to_excel_2.to_csv(save+name)


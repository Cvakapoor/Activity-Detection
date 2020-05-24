#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


f1=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1600_accel_phone.txt',sep='[,,;]',engine='python')
f2=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1601_accel_phone.txt',sep='[,,;]',engine='python')
f3=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1602_accel_phone.txt',sep='[,,;]',engine='python')
f4=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1603_accel_phone.txt',sep='[,,;]',engine='python')
f5=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1604_accel_phone.txt',sep='[,,;]',engine='python')
f6=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1605_accel_phone.txt',sep='[,,;]',engine='python')
f7=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1606_accel_phone.txt',sep='[,,;]',engine='python')
f8=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1607_accel_phone.txt',sep='[,,;]',engine='python')
f9=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1608_accel_phone.txt',sep='[,,;]',engine='python')
f10=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1609_accel_phone.txt',sep='[,,;]',engine='python')
f11=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1610_accel_phone.txt',sep='[,,;]',engine='python')
f12=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1611_accel_phone.txt',sep='[,,;]',engine='python')
f13=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1612_accel_phone.txt',sep='[,,;]',engine='python')
f14=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1613_accel_phone.txt',sep='[,,;]',engine='python')
f15=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1614_accel_phone.txt',sep='[,,;]',engine='python')
f16=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1615_accel_phone.txt',sep='[,,;]',engine='python')
f17=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1616_accel_phone.txt',sep='[,,;]',engine='python')
f18=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1617_accel_phone.txt',sep='[,,;]',engine='python')
f19=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1618_accel_phone.txt',sep='[,,;]',engine='python')
f20=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\accel\\data_1619_accel_phone.txt',sep='[,,;]',engine='python')
f1.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f2.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f3.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f4.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f5.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f6.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f7.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f8.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f9.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f10.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f11.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f12.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f13.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f14.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f15.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f16.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f17.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f18.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f19.columns=['subject_id','activity_label','timestamp','x','y','z','m']
f20.columns=['subject_id','activity_label','timestamp','x','y','z','m']
ph_accel_tra=pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20],axis=0,sort=False)
label_name = ph_accel_tra['activity_label'].map({'A': "WALKING", 'B':"JOGGING", 'C':"STAIRS", 'D':"SITTING", 'E':"STANDING", 'F':"TYPING",'G':'BRUSHING TEETH','H':'EATING SOUP','I':'EATING CHIPS','J':'EATING PASTA','K':'DRINKING FROM CUP','L':'EATING SANDWICH','M':'KICKING','O':'PLAYING CATCH','P':'DRIBBLING','Q':'WRITING','R':'CLAPPING','S':'FOLDING CLOTH'})
ph_accel_tra["activity_name"] = label_name


ph_accel_tra.columns=['subject_id','activity_label','timestamp','x_ph_accel','y_ph_accel','z_ph_accel','m',"activity_name"]
ph_accel_tra=ph_accel_tra.drop(['m'], axis = 1)
ph_accel_tra.head()


# In[3]:


ph_accel_tra.shape


# In[4]:


d1=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1600_gyro_phone.txt',sep='[,,;]',engine='python')
d2=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1601_gyro_phone.txt',sep='[,,;]',engine='python')
d3=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1602_gyro_phone.txt',sep='[,,;]',engine='python')
d4=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1603_gyro_phone.txt',sep='[,,;]',engine='python')
d5=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1604_gyro_phone.txt',sep='[,,;]',engine='python')
d6=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1605_gyro_phone.txt',sep='[,,;]',engine='python')
d7=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1606_gyro_phone.txt',sep='[,,;]',engine='python')
d8=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1607_gyro_phone.txt',sep='[,,;]',engine='python')
d9=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1608_gyro_phone.txt',sep='[,,;]',engine='python')
d10=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1609_gyro_phone.txt',sep='[,,;]',engine='python')
d11=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1610_gyro_phone.txt',sep='[,,;]',engine='python')
d12=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1611_gyro_phone.txt',sep='[,,;]',engine='python')
d13=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1612_gyro_phone.txt',sep='[,,;]',engine='python')
d14=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1613_gyro_phone.txt',sep='[,,;]',engine='python')
d15=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1614_gyro_phone.txt',sep='[,,;]',engine='python')
d16=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1615_gyro_phone.txt',sep='[,,;]',engine='python')
d17=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1616_gyro_phone.txt',sep='[,,;]',engine='python')
d18=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1617_gyro_phone.txt',sep='[,,;]',engine='python')
d19=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1618_gyro_phone.txt',sep='[,,;]',engine='python')
d20=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\phone\\gyro\\data_1619_gyro_phone.txt',sep='[,,;]',engine='python')
d1.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d2.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d3.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d4.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d5.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d6.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d7.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d8.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d9.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d10.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d11.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d12.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d13.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d14.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d15.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d16.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d17.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d18.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d19.columns=['subject_id','activity_label','timestamp','x','y','z','m']
d20.columns=['subject_id','activity_label','timestamp','x','y','z','m']
ph_gyro_tra=pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20],axis=0,sort=False)
label_name = ph_gyro_tra['activity_label'].map({'A': "WALKING", 'B':"JOGGING", 'C':"STAIRS", 'D':"SITTING", 'E':"STANDING", 'F':"TYPING",'G':'BRUSHING TEETH','H':'EATING SOUP','I':'EATING CHIPS','J':'EATING PASTA','K':'DRINKING FROM CUP','L':'EATING SANDWICH','M':'KICKING','O':'PLAYING CATCH','P':'DRIBBLING','Q':'WRITING','R':'CLAPPING','S':'FOLDING CLOTH'})
ph_gyro_tra["activity_name"] = label_name

ph_gyro_tra.columns=['subject_id','activity_label','timestamp','x_ph_gyro','y_ph_gyro','z_ph_gyro','m',"activity_name"]
ph_gyro_tra=ph_gyro_tra.drop(['m'], axis = 1)
ph_gyro_tra.head()


# In[5]:


ph_gyro_tra.shape


# In[6]:


a1=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1600_accel_watch.txt',sep='[,,;]',engine='python')
a2=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1601_accel_watch.txt',sep='[,,;]',engine='python')
a3=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1602_accel_watch.txt',sep='[,,;]',engine='python')
a4=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1603_accel_watch.txt',sep='[,,;]',engine='python')
a5=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1604_accel_watch.txt',sep='[,,;]',engine='python')
a6=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1605_accel_watch.txt',sep='[,,;]',engine='python')
a7=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1606_accel_watch.txt',sep='[,,;]',engine='python')
a8=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1607_accel_watch.txt',sep='[,,;]',engine='python')
a9=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1608_accel_watch.txt',sep='[,,;]',engine='python')
a10=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1609_accel_watch.txt',sep='[,,;]',engine='python')
a11=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1610_accel_watch.txt',sep='[,,;]',engine='python')
a12=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1611_accel_watch.txt',sep='[,,;]',engine='python')
a13=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1612_accel_watch.txt',sep='[,,;]',engine='python')
a14=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1613_accel_watch.txt',sep='[,,;]',engine='python')
a15=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1614_accel_watch.txt',sep='[,,;]',engine='python')
a16=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1615_accel_watch.txt',sep='[,,;]',engine='python')
a17=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1616_accel_watch.txt',sep='[,,;]',engine='python')
a18=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1617_accel_watch.txt',sep='[,,;]',engine='python')
a19=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1618_accel_watch.txt',sep='[,,;]',engine='python')
a20=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\accel\\data_1619_accel_watch.txt',sep='[,,;]',engine='python')
a1.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a2.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a3.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a4.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a5.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a6.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a7.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a8.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a9.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a10.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a11.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a12.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a13.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a14.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a15.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a16.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a17.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a18.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a19.columns=['subject_id','activity_label','timestamp','x','y','z','m']
a20.columns=['subject_id','activity_label','timestamp','x','y','z','m']
wat_accel_tra=pd.concat([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20],axis=0,sort=False)
label_name = wat_accel_tra['activity_label'].map({'A': "WALKING", 'B':"JOGGING", 'C':"STAIRS", 'D':"SITTING", 'E':"STANDING", 'F':"TYPING",'G':'BRUSHING TEETH','H':'EATING SOUP','I':'EATING CHIPS','J':'EATING PASTA','K':'DRINKING FROM CUP','L':'EATING SANDWICH','M':'KICKING','O':'PLAYING CATCH','P':'DRIBBLING','Q':'WRITING','R':'CLAPPING','S':'FOLDING CLOTH'})
wat_accel_tra["activity_name"] = label_name

wat_accel_tra.columns=['subject_id','activity_label','timestamp','x_wt_accel','y_wt_accel','z_wt_accel','m',"activity_name"]
wat_accel_tra=wat_accel_tra.drop(['m'], axis = 1)
wat_accel_tra.head()


# In[7]:


wat_accel_tra.shape


# In[8]:


s1=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1600_gyro_watch.txt',sep='[,,;]',engine='python')
s2=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1601_gyro_watch.txt',sep='[,,;]',engine='python')
s3=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1602_gyro_watch.txt',sep='[,,;]',engine='python')
s4=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1603_gyro_watch.txt',sep='[,,;]',engine='python')
s5=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1604_gyro_watch.txt',sep='[,,;]',engine='python')
s6=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1605_gyro_watch.txt',sep='[,,;]',engine='python')
s7=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1606_gyro_watch.txt',sep='[,,;]',engine='python')
s8=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1607_gyro_watch.txt',sep='[,,;]',engine='python')
s9=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1608_gyro_watch.txt',sep='[,,;]',engine='python')
s10=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1609_gyro_watch.txt',sep='[,,;]',engine='python')
s11=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1610_gyro_watch.txt',sep='[,,;]',engine='python')
s12=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1611_gyro_watch.txt',sep='[,,;]',engine='python')
s13=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1612_gyro_watch.txt',sep='[,,;]',engine='python')
s14=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1613_gyro_watch.txt',sep='[,,;]',engine='python')
s15=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1614_gyro_watch.txt',sep='[,,;]',engine='python')
s16=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1615_gyro_watch.txt',sep='[,,;]',engine='python')
s17=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1616_gyro_watch.txt',sep='[,,;]',engine='python')
s18=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1617_gyro_watch.txt',sep='[,,;]',engine='python')
s19=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1618_gyro_watch.txt',sep='[,,;]',engine='python')
s20=pd.read_csv('C:\\Users\\DELL1\\Desktop\\train\\watch\\gyro\\data_1619_gyro_watch.txt',sep='[,,;]',engine='python')
s1.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s2.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s3.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s4.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s5.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s6.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s7.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s8.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s9.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s10.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s11.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s12.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s13.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s14.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s15.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s16.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s17.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s18.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s19.columns=['subject_id','activity_label','timestamp','x','y','z','m']
s20.columns=['subject_id','activity_label','timestamp','x','y','z','m']
wat_gyro_tra=pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20],axis=0,sort=False)
label_name = wat_gyro_tra['activity_label'].map({'A': "WALKING", 'B':"JOGGING", 'C':"STAIRS", 'D':"SITTING", 'E':"STANDING", 'F':"TYPING",'G':'BRUSHING TEETH','H':'EATING SOUP','I':'EATING CHIPS','J':'EATING PASTA','K':'DRINKING FROM CUP','L':'EATING SANDWICH','M':'KICKING','O':'PLAYING CATCH','P':'DRIBBLING','Q':'WRITING','R':'CLAPPING','S':'FOLDING CLOTH'})
wat_gyro_tra["activity_name"] = label_name

wat_gyro_tra.columns=['subject_id','activity_label','timestamp','x_wt_gyro','y_wt_gyro','z_wt_gyro','m',"activity_name"]
wat_gyro_tra=wat_gyro_tra.drop(['m'], axis = 1)
wat_gyro_tra.head()


# In[9]:


wat_gyro_tra.shape


# In[10]:


train=pd.concat([ph_accel_tra,ph_gyro_tra,wat_accel_tra,wat_gyro_tra],axis=0,sort=False)
train.head()


# In[11]:


train.tail()


# In[12]:


X_train = train.drop(["subject_id", "activity_label","timestamp", "activity_name"], axis = 1)
y_train = train["activity_label"]


# In[13]:


from sklearn.preprocessing import Imputer
imp= Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(X_train)
X_train=imp.transform(X_train)
print(X_train[:10])


# In[14]:


X_train.shape


# In[15]:


q1=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1620_accel_phone.txt',sep='[,,;]',engine='python')
q2=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1621_accel_phone.txt',sep='[,,;]',engine='python')
q3=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1622_accel_phone.txt',sep='[,,;]',engine='python')
q4=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1623_accel_phone.txt',sep='[,,;]',engine='python')
q5=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1624_accel_phone.txt',sep='[,,;]',engine='python')
q6=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1625_accel_phone.txt',sep='[,,;]',engine='python')
q7=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1626_accel_phone.txt',sep='[,,;]',engine='python')
q8=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1627_accel_phone.txt',sep='[,,;]',engine='python')
q9=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1628_accel_phone.txt',sep='[,,;]',engine='python')
q10=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1629_accel_phone.txt',sep='[,,;]',engine='python')
q11=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1630_accel_phone.txt',sep='[,,;]',engine='python')
q12=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1631_accel_phone.txt',sep='[,,;]',engine='python')
q13=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1632_accel_phone.txt',sep='[,,;]',engine='python')
q14=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\accel\\data_1633_accel_phone.txt',sep='[,,;]',engine='python')
q1.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q2.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q3.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q4.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q5.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q6.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q7.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q8.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q9.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q10.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q11.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q12.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q13.columns=['subject_id','activity_label','timestamp','x','y','z','m']
q14.columns=['subject_id','activity_label','timestamp','x','y','z','m']
ph_accel_tes=pd.concat([q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14],axis=0,sort=False)
label_name = ph_accel_tes['activity_label'].map({'A': "WALKING", 'B':"JOGGING", 'C':"STAIRS", 'D':"SITTING", 'E':"STANDING", 'F':"TYPING",'G':'BRUSHING TEETH','H':'EATING SOUP','I':'EATING CHIPS','J':'EATING PASTA','K':'DRINKING FROM CUP','L':'EATING SANDWICH','M':'KICKING','O':'PLAYING CATCH','P':'DRIBBLING','Q':'WRITING','R':'CLAPPING','S':'FOLDING CLOTH'})
ph_accel_tes["activity_name"] = label_name

ph_accel_tes.columns=['subject_id','activity_label','timestamp','x_ph_accel','y_ph_accel','z_ph_accel','m',"activity_name"]
ph_accel_tes=ph_accel_tes.drop(['m'], axis = 1)

ph_accel_tes.head()


# In[16]:


z1=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1620_gyro_phone.txt',sep='[,,;]',engine='python')
z2=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1621_gyro_phone.txt',sep='[,,;]',engine='python')
z3=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1622_gyro_phone.txt',sep='[,,;]',engine='python')
z4=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1623_gyro_phone.txt',sep='[,,;]',engine='python')
z5=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1624_gyro_phone.txt',sep='[,,;]',engine='python')
z6=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1625_gyro_phone.txt',sep='[,,;]',engine='python')
z7=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1626_gyro_phone.txt',sep='[,,;]',engine='python')
z8=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1627_gyro_phone.txt',sep='[,,;]',engine='python')
z9=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1628_gyro_phone.txt',sep='[,,;]',engine='python')
z10=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1629_gyro_phone.txt',sep='[,,;]',engine='python')
z11=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1630_gyro_phone.txt',sep='[,,;]',engine='python')
z12=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1631_gyro_phone.txt',sep='[,,;]',engine='python')
z13=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1632_gyro_phone.txt',sep='[,,;]',engine='python')
z14=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\phone\\gyro\\data_1633_gyro_phone.txt',sep='[,,;]',engine='python')
z1.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z2.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z3.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z4.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z5.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z6.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z7.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z8.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z9.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z10.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z11.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z12.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z13.columns=['subject_id','activity_label','timestamp','x','y','z','m']
z14.columns=['subject_id','activity_label','timestamp','x','y','z','m']
ph_gyro_tes=pd.concat([z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14],axis=0,sort=False)
label_name = ph_gyro_tes['activity_label'].map({'A': "WALKING", 'B':"JOGGING", 'C':"STAIRS", 'D':"SITTING", 'E':"STANDING", 'F':"TYPING",'G':'BRUSHING TEETH','H':'EATING SOUP','I':'EATING CHIPS','J':'EATING PASTA','K':'DRINKING FROM CUP','L':'EATING SANDWICH','M':'KICKING','O':'PLAYING CATCH','P':'DRIBBLING','Q':'WRITING','R':'CLAPPING','S':'FOLDING CLOTH'})
ph_gyro_tes["activity_name"] = label_name

ph_gyro_tes.columns=['subject_id','activity_label','timestamp','x_ph_gyro','y_ph_gyro','z_ph_gyro','m',"activity_name"]
ph_gyro_tes=ph_gyro_tes.drop(['m'], axis = 1)

ph_gyro_tes.head()


# In[17]:


w1=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1620_accel_watch.txt',sep='[,,;]',engine='python')
w2=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1621_accel_watch.txt',sep='[,,;]',engine='python')
w3=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1622_accel_watch.txt',sep='[,,;]',engine='python')
w4=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1623_accel_watch.txt',sep='[,,;]',engine='python')
w5=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1624_accel_watch.txt',sep='[,,;]',engine='python')
w6=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1625_accel_watch.txt',sep='[,,;]',engine='python')
w7=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1626_accel_watch.txt',sep='[,,;]',engine='python')
w8=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1627_accel_watch.txt',sep='[,,;]',engine='python')
w9=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1628_accel_watch.txt',sep='[,,;]',engine='python')
w10=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1629_accel_watch.txt',sep='[,,;]',engine='python')
w11=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1630_accel_watch.txt',sep='[,,;]',engine='python')
w12=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1631_accel_watch.txt',sep='[,,;]',engine='python')
w13=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1632_accel_watch.txt',sep='[,,;]',engine='python')
w14=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\accel\\data_1633_accel_watch.txt',sep='[,,;]',engine='python')
w1.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w2.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w3.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w4.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w5.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w6.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w7.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w8.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w9.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w10.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w11.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w12.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w13.columns=['subject_id','activity_label','timestamp','x','y','z','m']
w14.columns=['subject_id','activity_label','timestamp','x','y','z','m']
wat_accel_tes=pd.concat([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14],axis=0,sort=False)
label_name = wat_accel_tes['activity_label'].map({'A': "WALKING", 'B':"JOGGING", 'C':"STAIRS", 'D':"SITTING", 'E':"STANDING", 'F':"TYPING",'G':'BRUSHING TEETH','H':'EATING SOUP','I':'EATING CHIPS','J':'EATING PASTA','K':'DRINKING FROM CUP','L':'EATING SANDWICH','M':'KICKING','O':'PLAYING CATCH','P':'DRIBBLING','Q':'WRITING','R':'CLAPPING','S':'FOLDING CLOTH'})
wat_accel_tes["activity_name"] = label_name

wat_accel_tes.columns=['subject_id','activity_label','timestamp','x_wt_accel','y_wt_accel','z_wt_accel','m',"activity_name"]
wat_accel_tes=wat_accel_tes.drop(['m'], axis = 1)

wat_accel_tes.head()


# In[18]:


wat_accel_tes.shape


# In[19]:


x1=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1620_gyro_watch.txt',sep='[,,;]',engine='python')
x2=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1621_gyro_watch.txt',sep='[,,;]',engine='python')
x3=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1622_gyro_watch.txt',sep='[,,;]',engine='python')
x4=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1623_gyro_watch.txt',sep='[,,;]',engine='python')
x5=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1624_gyro_watch.txt',sep='[,,;]',engine='python')
x6=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1625_gyro_watch.txt',sep='[,,;]',engine='python')
x7=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1626_gyro_watch.txt',sep='[,,;]',engine='python')
x8=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1627_gyro_watch.txt',sep='[,,;]',engine='python')
x9=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1628_gyro_watch.txt',sep='[,,;]',engine='python')
x10=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1629_gyro_watch.txt',sep='[,,;]',engine='python')
x11=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1630_gyro_watch.txt',sep='[,,;]',engine='python')
x12=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1631_gyro_watch.txt',sep='[,,;]',engine='python')
x13=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1632_gyro_watch.txt',sep='[,,;]',engine='python')
x14=pd.read_csv('C:\\Users\\DELL1\\Desktop\\test\\watch\\gyro\\data_1633_gyro_watch.txt',sep='[,,;]',engine='python')
x1.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x2.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x3.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x4.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x5.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x6.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x7.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x8.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x9.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x10.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x11.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x12.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x13.columns=['subject_id','activity_label','timestamp','x','y','z','m']
x14.columns=['subject_id','activity_label','timestamp','x','y','z','m']
wat_gyro_tes=pd.concat([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14],axis=0,sort=False)
label_name = wat_gyro_tes['activity_label'].map({'A': "WALKING", 'B':"JOGGING", 'C':"STAIRS", 'D':"SITTING", 'E':"STANDING", 'F':"TYPING",'G':'BRUSHING TEETH','H':'EATING SOUP','I':'EATING CHIPS','J':'EATING PASTA','K':'DRINKING FROM CUP','L':'EATING SANDWICH','M':'KICKING','O':'PLAYING CATCH','P':'DRIBBLING','Q':'WRITING','R':'CLAPPING','S':'FOLDING CLOTH'})
wat_gyro_tes["activity_name"] = label_name

wat_gyro_tes.columns=['subject_id','activity_label','timestamp','x_wt_gyro','y_wt_gyro','z_wt_gyro','m',"activity_name"]
wat_gyro_tes=wat_gyro_tes.drop(['m'], axis = 1)

wat_gyro_tes.head()


# In[20]:


test=pd.concat([ph_accel_tes,ph_gyro_tes,wat_accel_tes,wat_gyro_tes],axis=0,sort=False)
test.head()


# In[21]:


test.tail()


# In[22]:


X_test = test.drop(["subject_id", "activity_label","timestamp", "activity_name"], axis = 1)
y_test = test["activity_label"]


# In[23]:


from sklearn.preprocessing import Imputer
imp= Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(X_test)
X_test=imp.transform(X_test)
print(X_train[:10])
X_test.shape


# In[59]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train[:3600],y_train[:3600])
y_pred=knn.predict(X_test[:3600])
accuracy=(accuracy_score(y_test[:3600],y_pred[:3600]))*100
print('Accuracy is :' +str(accuracy)+'%')
print(confusion_matrix(y_test[:3600],y_pred[:3600]))
KNN_accuracy= str(accuracy)+'%'


# In[61]:


from sklearn.svm import SVC
svc=SVC(kernel='rbf')
svc.fit(X_train[:3600],y_train[:3600])
y_pred=svc.predict(X_test[:3600])
accuracy=(accuracy_score(y_test[:3600],y_pred[:3600]))*100
print('Accuracy is :' +str(accuracy)+'%')
print(confusion_matrix(y_test[:3600],y_pred[:3600]))
SVC_accuracy= str(accuracy)+'%'


# In[62]:


logreg=LogisticRegression(multi_class='ovr')
logreg.fit(X_train[:3600],y_train[:3600])
y_pred=logreg.predict(X_test[:3600])
accuracy=(accuracy_score(y_test[:3600],y_pred[:3600]))*100
print('Accuracy is :' +str(accuracy)+'%')
print(confusion_matrix(y_test[:3600],y_pred[:3600]))
LOGREG_accuracy= str(accuracy)+'%'


# In[63]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train[:3600],y_train[:3600])
y_pred=dtc.predict(X_test[:3600])
accuracy=(accuracy_score(y_test[:3600],y_pred[:3600]))*100
print('Accuracy is :' +str(accuracy)+'%')
print(confusion_matrix(y_test[:3600],y_pred[:3600]))
DTC_accuracy= str(accuracy)+'%'


# In[64]:


rfc=RandomForestClassifier()
rfc.fit(X_train[:3600],y_train[:3600])
y_pred=rfc.predict(X_test[:3600])
accuracy=(accuracy_score(y_test[:3600],y_pred[:3600]))*100
print('Accuracy is :' +str(accuracy)+'%')
print(confusion_matrix(y_test[:3600],y_pred[:3600]))
RFC_accuracy= str(accuracy)+'%'


# In[73]:


accuracy= pd.DataFrame({'Classifier':['KNeighborsClassifier','SVC','LogisticRegression','DecisionTreeClassifier','RandomForestClassifier'],'Accuracy':[KNN_accuracy,SVC_accuracy,LOGREG_accuracy,DTC_accuracy,RFC_accuracy]})
print(accuracy)


# In[ ]:





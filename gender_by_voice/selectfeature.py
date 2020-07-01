#"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

voice_data=pd.read_csv('voice.csv')
x=voice_data.iloc[:,:-1]
y=voice_data.iloc[:,-1]
feat_labels = voice_data.columns[:-1]
y = LabelEncoder().fit_transform(y)
imp=SimpleImputer(missing_values=0,strategy='mean')
x=imp.fit_transform(x) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

#"""
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

voice_data=pd.read_csv('voice.csv')
male=voice_data.iloc[:1583,:]
male_x1=male['IQR']
male_x2=male['meanfun']
 
female=voice_data.iloc[1584:,:]
female_x1=female['IQR']
female_x2=female['meanfun']
plt.figure()
plt.scatter(male_x1,male_x2,c='b',alpha=0.5,label='male')
plt.scatter(female_x1,female_x2,c='r',alpha=0.5,marker="p",label='female')
plt.xlabel('IQR')
plt.ylabel('meanfun')
plt.legend(loc='upper right')    
 
plt.show()
"""
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import pandas as pd
from sklearn.impute import SimpleImputer 


voice_data=pd.read_csv('voice.csv')
voice_data=voice_data[['meanfun','IQR','Q25','label']]
x=voice_data.iloc[:,:-1]
imp=SimpleImputer(missing_values=0,strategy='mean')
x=imp.fit_transform(x) 





voice_data=pd.read_csv('voice.csv')
voice_data=voice_data.iloc[:,:-1]
imp=SimpleImputer(missing_values=0,strategy='mean')
voice_data=imp.fit_transform(voice_data)
male=x[:1583,:]
male_x1=male[:,2]
male_x2=male[:,0]
male_x3=male[:,1] 

female=x[1584:,:]
female_x1=female[:,2]
female_x2=female[:,0]
female_x3=female[:,1]
 
 
# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(male_x1, male_x2, male_x3, c='r', label='随机点')
ax.scatter(female_x1, female_x2, female_x3, c='g', label='随机点')
 
 
# 绘制图例
ax.legend(loc='best')
 
 
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('IQR')
ax.set_ylabel('meanfun')
ax.set_xlabel('Q25')
 
 
# 
plt.show()
"""
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

voice_data=pd.read_csv('voice.csv')
voice_data=voice_data[['meanfun','IQR','Q25','label']]
"""
















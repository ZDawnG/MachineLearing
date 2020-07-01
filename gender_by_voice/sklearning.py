import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 


voice_data=pd.read_csv('voice.csv')
voice_data=voice_data[['meanfun','IQR','Q25','label']]
x=voice_data.iloc[:,:-1]
y=voice_data.iloc[:,-1]
y = LabelEncoder().fit_transform(y)
imp=SimpleImputer(missing_values=0,strategy='mean')
x=imp.fit_transform(x) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

predictionrate=[]

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_predict = mnb.predict(x_test)
print('MultinomialNB准确率：', mnb.score(x_test, y_test))
print(classification_report(y_test, y_predict))
predictionrate.append(mnb.score(x_test, y_test))

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_predict = gnb.predict(x_test)
print('GaussianNB准确率：', gnb.score(x_test, y_test))
print(classification_report(y_test, y_predict))
predictionrate.append(gnb.score(x_test, y_test))

scaler1 = StandardScaler()
scaler1.fit(x_train)
x_train = scaler1.transform(x_train)
x_test = scaler1.transform(x_test)

svc=SVC(C=10.0, kernel='rbf', probability=True) 
svc.fit(x_train,y_train) 
y_predict = svc.predict(x_test)
print('SVM准确率：', svc.score(x_test, y_test))
print(classification_report(y_test, y_predict))
predictionrate.append(svc.score(x_test, y_test))

logistic=LogisticRegression(max_iter=10000) 
logistic.fit(x_train,y_train)
y_predict = svc.predict(x_test)
print('logistic准确率：', logistic.score(x_test, y_test))
print(classification_report(y_test, y_predict))
predictionrate.append(logistic.score(x_test, y_test))

cart=DecisionTreeClassifier() 
cart.fit(x_train,y_train)
y_predict = cart.predict(x_test)
print('DecisionTree准确率：', cart.score(x_test, y_test))
print(classification_report(y_test, y_predict))
predictionrate.append(cart.score(x_test, y_test))


from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt


name=['MultinomialNB','GaussianNB','SVM','logistic','DecisionTree']
plt.bar(range(len(predictionrate)), predictionrate,color='rgb',tick_label=name)

plt.figure()


###############画cart的ROC-AUC曲线########################
prob_predict_y_validation_cart = cart.predict_proba(x_test)#给出带有概率值的结果，每个点所有label的概率和为1
predictions_validation_cart = prob_predict_y_validation_cart[:, 1]  
fpr_cart, tpr_cart, _ = roc_curve(y_test, predictions_validation_cart) 
roc_auc_cart = auc(fpr_cart, tpr_cart)  
plt.plot(fpr_cart, tpr_cart, 'b', label='cart = %0.2f' % roc_auc_cart) 

###############画svm的ROC-AUC曲线########################

prob_predict_y_validation_svm = svc.predict_proba(x_test)#给出带有概率值的结果，每个点所有label的概率和为1
predictions_validation_svm = prob_predict_y_validation_svm[:, 1]  
fpr_svm, tpr_svm, _ = roc_curve(y_test, predictions_validation_svm) 
roc_auc_svm = auc(fpr_svm, tpr_svm)  
plt.plot(fpr_svm, tpr_svm, 'm', label='svm = %0.2f' % roc_auc_svm) 

###############画logistic的ROC-AUC曲线########################

prob_predict_y_validation_logistic = logistic.predict_proba(x_test)#给出带有概率值的结果，每个点所有label的概率和为1
predictions_validation_logistic = prob_predict_y_validation_logistic[:, 1]  
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, predictions_validation_logistic) 
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)  
plt.plot(fpr_logistic, tpr_logistic, 'g', label='logistic = %0.2f' % roc_auc_logistic) 

###############################roc auc公共设置##################################

plt.title('ROC Validation')  
plt.legend(loc='lower right')  
plt.plot([0, 1], [0, 1], 'r--')  
plt.xlim([0, 1])  
plt.ylim([0, 1])  
plt.ylabel('True Positive Rate')  
plt.xlabel('False Positive Rate') 



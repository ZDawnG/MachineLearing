import csv
import numpy as np
import matplotlib.pyplot as plt

class GaussianNativeBayes():
    '''高斯朴素贝叶斯分类器'''
 
    def __init__(self):
 
        self._X_train = None
        self._y_train = None
        self._classes = None
        self._priorlist = None
        self._meanmat = None
        self._varmat = None
#加载文件，切分数据集
    def load_data_set(self,file_name):
        """
        :param file_name: 文件名字
        :return
    
        train_mat：离散化的训练数据集
        train_classes： 训练数据集所属的分类
        test_mat：离散化的测试数据集
        test_classes：测试数据集所述的分类
        label_name：特征的名称
        """
        data_mat = []
        with open(file_name) as file_obj:
            voice_reader = csv.DictReader(file_obj)
            list_class = []
            # 文件头
            label_name = list(voice_reader.fieldnames)
            num = len(label_name) - 1
    
            for line in voice_reader.reader:
                data_mat.append(line[:num])
                gender = 1 if line[-1] == 'male' else 0
                list_class.append(gender)
    
            # 求每一个特征的平均值
            data_mat = np.array(data_mat).astype(float)
            count_vector = np.count_nonzero(data_mat, axis=0)
            sum_vector = np.sum(data_mat, axis=0)
            mean_vector = sum_vector / count_vector
    
            # 数据缺失的地方 用 平均值填充
            for row in range(len(data_mat)):
                for col in range(num):
                    if data_mat[row][col] == 0.0:
                        data_mat[row][col] = mean_vector[col]
            
            # 将数据连续型的特征值离散化处理
            min_vector = data_mat.min(axis=0)
            max_vector = data_mat.max(axis=0)
            diff_vector = max_vector - min_vector
            diff_vector /= (divison-1)
    
            new_data_set = []
            for i in range(len(data_mat)):
                line = np.array((data_mat[i] - min_vector) / diff_vector).astype(int)
                new_data_set.append(line)
            """
            new_data_set = []
            for i in range(len(data_mat)):
                line = np.array(data_mat[i]).astype(int)
                new_data_set.append(line)
            """
            # 随机划分数据集为训练集 和 测试集
            test_set = list(range(len(new_data_set)))
            train_set = []
            for i in range(2200):
                random_index = int(np.random.uniform(0, len(test_set)))
                train_set.append(test_set[random_index])
                del test_set[random_index]
    
            # 训练数据集
            train_mat = []
            train_classes = []
            for index in train_set:
                train_mat.append(new_data_set[index])
                train_classes.append(list_class[index])
    
            # 测试数据集
            test_mat = []
            test_classes = []
            for index in test_set:
                test_mat.append(new_data_set[index])
                test_classes.append(list_class[index])
        train_mat = np.array(train_mat)
        train_classes = np.array(train_classes)
        test_mat = np.array(test_mat)
        test_classes = np.array(test_classes)
        return train_mat, train_classes, test_mat, test_classes, label_name

#求先验概率 
    def fit(self, X_train, y_train):
        
        self._X_train = X_train
        self._y_train = y_train
        self._classes = np.unique(self._y_train)                       #  得到各个类别
        priorlist = []
        meanmat0 = np.zeros([1,20])
        varmat0 = np.zeros([1,20])
        for i, c in enumerate(self._classes):
            # 计算每个种类的平均值，方差，先验概率
            X_Index_c = self._X_train[np.where(self._y_train == c)]        # 属于某个类别的样本组成的“矩阵”
            priorlist.append(X_Index_c.shape[0] / self._X_train.shape[0])  # 计算类别的先验概率
            X_index_c_mean = np.mean(X_Index_c, axis=0, keepdims=True)     # 计算该类别下每个特征的均值，结果保持二维状态[[3 4 6 2 1]]
            X_index_c_var = np.var(X_Index_c, axis=0, keepdims=True)       # 方差
            meanmat0 = np.append(meanmat0, X_index_c_mean, axis=0)         # 各个类别下的特征均值矩阵罗成新的矩阵，每行代表一个类别。
            varmat0 = np.append(varmat0, X_index_c_var, axis=0)
        self._priorlist = priorlist
        self._meanmat = meanmat0[1:, :]                                    #除去开始多余的第一行
        self._varmat = varmat0[1:, :]

#预测类别 
    def predict(self,X_test):
        
        eps = 1e-10                                                        # 防止分母为0
        classof_X_test = []                                                #用于存放测试集中各个实例的所属类别
        for x_sample in X_test:
            matx_sample = np.tile(x_sample,(len(self._classes),1))         #将每个实例沿列拉长，行数为样本的类别数
            mat_numerator = np.exp(-(matx_sample - self._meanmat) ** 2 / (2 * self._varmat + eps))
            mat_denominator = np.sqrt(2 * np.pi * self._varmat + eps)
            list_log = np.sum(np.log(mat_numerator/mat_denominator),axis=1)# 每个类别下的类条件概率取对数后相加
            prior_class_x = list_log + np.log(self._priorlist)             # 加上类先验概率的对数
            prior_class_x_index = np.argmax(prior_class_x)                 # 取对数概率最大的索引
            classof_x = self._classes[prior_class_x_index]                 # 返回一个实例对应的类别
            classof_X_test.append(classof_x)
        return classof_X_test
 
#评价函数
    def score(self, X_predict, y_test):
        count = 0.0
        male_count = 0.0
        female_count = 0.0
        correct_male_count = 0.0
        correct_female_count = 0.0
        false_male_count = 0.0
        false_female_count = 0.0
        for i in range(len(X_predict)):
            if X_predict[i] == y_test[i]:
                if y_test[i]  == 1:
                    correct_male_count += 1
                    male_count += 1
                else:
                    correct_female_count += 1
                    female_count += 1
            else:
                if y_test[i]  == 1:
                    false_male_count += 1
                    male_count += 1
                else:
                    false_female_count += 1
                    female_count += 1
            count += 1 
            
        male_correct_rate = correct_male_count / male_count
        female_correct_rate = correct_female_count / female_count
        male_fail_rate = false_male_count / male_count
        female_fail_rate = false_female_count / female_count
        total_correct_rate = (correct_male_count+correct_female_count)/count
        print('male correct rate:   ', correct_male_count / male_count)
        print('female correct rate: ', correct_female_count / female_count)
        print('male fail rate:      ', false_male_count / male_count)
        print('female fail rate:    ', false_female_count / female_count)
        print('total correct rate:  ', (correct_male_count+correct_female_count)/count)
        
        return male_correct_rate,female_correct_rate,male_fail_rate,female_fail_rate,total_correct_rate
        


#测试函数
def test_bayes():
    file_name = 'voice.csv'
    gnb = GaussianNativeBayes()
    train_mat, train_classes, test_mat, test_classes, label_name = gnb.load_data_set(file_name)  
    gnb.fit(train_mat,train_classes)
    x_predict = gnb.predict(test_mat)   
    #print(nb.score(x_predict,test_classes))
    male_correct_rate,female_correct_rate,male_fail_rate,female_fail_rate,total_correct_rate = gnb.score(x_predict,test_classes)
    return male_correct_rate,female_correct_rate,male_fail_rate,female_fail_rate,total_correct_rate



divison = 30    #量化阶
n = 10

rate = np.zeros([1,5])
for i in range(n):
    t = np.zeros(5)
    t = test_bayes()
    rate = np.append(rate,[t],axis=0)
rate = rate[1:,:]
avr = np.sum(rate,axis = 0)
avr /= n
print('male correct average rate:   ', avr[0])
print('female correct average rate: ', avr[1])
print('male fail average rate:      ', avr[2])
print('female fail average rate:    ', avr[3])
print('total correct average rate:  ', avr[4])


lianghua =  np.zeros([1,5])
for d in range(10,60,2):
    divion = d
    rate = np.zeros([1,5])
    for i in range(n):
        t = np.zeros(5)
        t = test_bayes()
        rate = np.append(rate,[t],axis=0)
    rate = rate[1:,:]
    avr = np.sum(rate,axis = 0)
    avr /= n
    lianghua = np.append(lianghua,[avr],axis=0)
lianghua = lianghua[1:]
title=['male_correct_rate','female_correct_rate','male_fail_rate','female_fail_rate','total_correct_rate']
x = list(range(10,60,2))
for i in range(5):
    #plt.subplot2grid((2,3),(i//3,i%3))
    y = lianghua[:,i]
    #plt.plot(x,label='x')
    plt.title(title[i])
    plt.plot(x,y)
    plt.show()



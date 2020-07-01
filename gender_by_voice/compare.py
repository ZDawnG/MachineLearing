import csv
import numpy as np

class NaiveBayes():
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
    
            # 随机划分数据集为训练集 和 测试集
            test_set = list(range(len(new_data_set)))
            train_set = []
            for i in range(2000):
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
    
        return train_mat, train_classes, test_mat, test_classes, label_name


    def native_bayes(self,train_matrix, list_classes):
        """
        :param train_matrix: 训练样本矩阵
        :param list_classes: 训练样本分类向量
        :return:p_1_class 任一样本分类为1的概率  p_feature,p_1_feature 分别为给定类别的情况下所以特征所有取值的概率
        """
    
        # 训练样本个数
        num_train_data = len(train_matrix)
        num_feature = len(train_matrix[0])
        # 分类为1的样本占比
        p_1_class = sum(list_classes) / float(num_train_data)
        p_0_class = 1 - p_1_class
        n = divison
        list_classes_1 = []
        train_data_1 = []
        list_classes_0 = []
        train_data_0 = []
        
        for i in list(range(num_train_data)):
            if list_classes[i] == 1:
                list_classes_1.append(i)
                train_data_1.append(train_matrix[i])
            else:
                list_classes_0.append(i)
                train_data_0.append(train_matrix[i])
    
        # 分类为1 情况下的各特征的概率
        train_data_1 = np.matrix(train_data_1)
        p_1_feature = {}
        for i in list(range(num_feature)):
            feature_values = np.array(train_data_1[:, i]).flatten()
            # 避免某些特征值概率为0 影响总体概率，每个特征值最少个数为1
            feature_values = feature_values.tolist() + list(range(n))
            p = {}
            count = len(feature_values)
            for value in set(feature_values):
                p[value] = np.log(feature_values.count(value) / float(count))
            p_1_feature[i] = p
        
        # 分类为0 情况下的各特征的概率
        train_data_0 = np.matrix(train_data_0)
        p_0_feature = {}
        for i in list(range(num_feature)):
            feature_values = np.array(train_data_0[:, i]).flatten()
            # 避免某些特征值概率为0 影响总体概率，每个特征值最少个数为1
            feature_values = feature_values.tolist() + list(range(n))
            p = {}
            count = len(feature_values)
            for value in set(feature_values):
                p[value] = np.log(feature_values.count(value) / float(count))
            p_0_feature[i] = p
        
        # 所有分类下的各特征的概率
        p_feature = {}
        train_matrix = np.matrix(train_matrix)
        for i in list(range(num_feature)):
            feature_values = np.array(train_matrix[:, i]).flatten()
            feature_values = feature_values.tolist() + list(range(n))
            p = {}
            count = len(feature_values)
            for value in set(feature_values):
                p[value] = np.log(feature_values.count(value) / float(count))
            p_feature[i] = p
    
        return p_feature, p_1_feature, p_1_class, p_0_feature, p_0_class
    
    
    def classify_bayes(self,test_vector, p_feature, p_1_feature, p_1_class, p_0_feature, p_0_class):
        """
        :param test_vector: 要分类的测试向量
        :param p_feature: 所有分类的情况下特征所有取值的概率
        :param p_1_feature: 类别为1的情况下所有特征所有取值的概率
        :param p_1_class: 任一样本分类为1的概率
        :return: 1 表示男性 0 表示女性
        """
        # 计算每个分类的概率(概率相乘取对数 = 概率各自对数相加)
        sum = 0.0
        for i in list(range(len(test_vector))):
            sum += p_1_feature[i][test_vector[i]]
            sum -= p_feature[i][test_vector[i]]
        p1 = sum + np.log(p_1_class)
        p0 = 1 - p1
        sum = 0.0
        for i in list(range(len(test_vector))):
            sum += p_0_feature[i][test_vector[i]]
            sum -= p_feature[i][test_vector[i]]
        p0 = sum + np.log(p_0_class)
        if p1 > p0:
            return 1
        else:
            return 0

class GaussianNativeBayes():
    '''高斯朴素贝叶斯分类器'''
 
    def __init__(self):
 
        self._X_train = None
        self._y_train = None
        self._classes = None
        self._priorlist = None
        self._meanmat = None
        self._varmat = None

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
            for i in range(2000):
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
 
    def score(self, X_predict, y_test):
        
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
        print(correct_male_count / male_count)
        print(correct_female_count / female_count)
        print(false_male_count / male_count)
        print(false_female_count / female_count)



def test_bayes():

    file_name = 'voice.csv'
    nb = NaiveBayes()
    gnb = GaussianNativeBayes()
    train_mat, train_classes, test_mat, test_classes, label_name = gnb.load_data_set(file_name)  
    gnb.fit(train_mat,train_classes)
    x_predict = gnb.predict(test_mat)       
    p_feature, p_1_feature, p_1_class, p_0_feature, p_0_class = nb.native_bayes(train_mat, train_classes)
    
    count = 0.0
    male_count = 0.0
    female_count = 0.0
    correct_male_count = 0.0
    correct_female_count = 0.0
    false_male_count = 0.0
    false_female_count = 0.0
    
    for i in list(range(len(test_mat))):
        test_vector = test_mat[i]
        result = nb.classify_bayes(test_vector, p_feature, p_1_feature, p_1_class, p_0_feature, p_0_class)
        result_g = x_predict[i]
        if (result == 1) and (test_classes[i] == 1):
            correct_male_count += 1
            male_count += 1
        
        if (result == 0) and (test_classes[i] == 1):
            false_male_count += 1
            male_count += 1
                     
        if (test_classes[i] == 0) and ((result == test_classes[i]) or (result_g == test_classes[i])) :
            correct_female_count += 1
            female_count += 1        
        else:
            false_female_count += 1
            female_count += 1
        count += 1
    print(correct_male_count / male_count)
    print(correct_female_count / female_count)
    print(false_male_count / male_count)
    print(false_female_count / female_count)
    print((correct_male_count+correct_female_count)/count)



def test_bayess():

    file_name = 'voice.csv'
    nb = NaiveBayes()
    gnb = GaussianNativeBayes()
    train_mat, train_classes, test_mat, test_classes, label_name = gnb.load_data_set(file_name)

    p_feature, p_1_feature, p_1_class, p_0_feature, p_0_class = nb.native_bayes(train_mat, train_classes)


    
    gnb.fit(train_mat,train_classes)
    x_predict = gnb.predict(test_mat)       
    
    count = 0.0
    male_count = 0.0
    female_count = 0.0
    correct_male_count = 0.0
    correct_female_count = 0.0
    false_male_count = 0.0
    false_female_count = 0.0
    for i in list(range(len(test_mat))):
        test_vector = test_mat[i]
        result = nb.classify_bayes(test_vector, p_feature, p_1_feature, p_1_class, p_0_feature, p_0_class)
        if result == test_classes[i]:
            if test_classes[i] == 1:
                correct_male_count += 1
                male_count += 1
            if x_predict[i] == 0 or result ==0:
                correct_female_count += 1
                female_count += 1
        else:
            if test_classes[i] == 1:
                false_male_count += 1
                male_count += 1
            else:
                false_female_count += 1
                female_count += 1
        count += 1
    print(correct_male_count / male_count)
    print(correct_female_count / female_count)
    print(false_male_count / male_count)
    print(false_female_count / female_count)
    print((correct_male_count+correct_female_count)/count)
    
    
    
divison = 10
test_bayess()
### Scikit-Learn的使用手册
---
<font size=4>[1.例子](#1)</font><br>
<font size=4>[2.数据加载](#2)</font><br>
<font size=4>[3.训练和测试数据](#3)</font><br>
<font size=4>[4.数据预处理](#4)</font><br>
<font size=4>[5.模型创建](#5)</font><br>
<font size=4>[6.模型拟合](#6)</font><br>
<font size=4>[7.模型预测](#7)</font><br>
<font size=4>[8.模型性能评估](#8)</font><br>
<font size=4>[9.模型调整](#9)</font><br>

---

<h4 id="1">1.例子Demo</h4>

  ```python
  from sklearn import neighbors,datasets,preprocessing
  from sklearn.model_selection import train_test_split
  from sklearn.mettics import accuracy_score
  iris = datasets.load_iris()
  X,y = iris.data[:,:2], iris.target
  X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33)
  scaler = preprocessing.StandardScaler().fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  knn = neighbors.KNeighborsClassifier(n_neighbors=5)
  knn.fit(X_train,y_train)
  y_pred = knn.predict(X_test)
  accuracy_score(y_test,y_pred)
  ```

<h4 id="2">2.数据加载</h4>

  ```python
  import numpy as np
  X = np.random.random((10,5))
  y = np.array(['M','M','F','F','M','F','M','M','F','F','F'])
  X[X<0.7] = 0
  ```

<h4 id="3">3.训练和测试数据</h4>

  ```python
  from sklearn.model_selection import  train_test_split
  X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
  ```

<h4 id="4">4.数据预处理</h4>

  ```python
  # 标准话Standardization
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler().fit(X_train)
  standardized_X = scaler.transform(X_train)
  standardized_X_test = scaler.transform(X_test)
  # 正则化、归一化Normalization
  from sklearn.preprocessing import Normalizer
  scaler = Normalizer().fit(X_train)
  normalized_X = scaler.trainsform(X_train)
  normalized_X_test = scaler.transform(X_test)
  # 二值化Binarization
  from sklearn.preprocessing import Binarizer
  binarizer = Binarizer(threshold=0.0).fit(X)
  binary_X = binarizer.transform(X)
  # 类属特征Encoding Categorical Features
  from sklearn.preprocessing import LabelEncoder
  enc = LaberEncoder()
  y = enc.fit_transform(y)
  # 填充缺失值 Imputing Missing values
  from sklearn.preprocessing import Imputer
  imp = Imputer(missing_values=0,strategy='mean',axis=0)
  imp.fit_transform(X_train)
  # 多项式特征 Generating Polynomial Features
  from sklearn.preprocessing import PolynomialFeatures
  poly = PolynomialFeatures(5)
  poly.fit_transform(X)
  ```

<h4 id="5">5.模型创建</h4>

  ```python
  # 监督学习Supervised Learning Estimators
  # 线性回归Linear Regression
  from sklearn.linear_model import LinearRegression
  lr = LinearRegression(normalize= True)
  # 支持向量机 Support Vector Machines(SVM)
  from sklearn.svm import SVC
  svc = SVC(kernel = 'linear')
  # 朴素贝叶斯 Naive Bayes
  from sklearn.naive_bayes import GaussianNB
  gnb = GaussianNB()
  # K最临近节点算法 KNN
  from sklearn import n_neighbors
  knn = neighbors.KNeighborClassifier(n_neighbors=5)
  # 非监督学习Unsupervised Learning Estimators
  # 主成分分析
  from sklearn.decomposition import PCA
  pca = PCA(n_components=0.95)
  # K-Means算法 聚类算法的一种
  from sklearn.cluster import KMeans
  k_means = KMeans(n_clusters=3,random_state=0)
  ```

<h4 id="6">6.模型拟合</h4>

  ```python
  # 监督学习Supervised Learning
  lr.fir(X,y)
  knn.fit(X_train,y_train)
  svc.fit(X_train,y_train)
  # 非监督学习
  k_means.fit(X_train)
  pca_model = pca.fit_transform(X_train)
  ```

<h4 id="7">7.模型预测</h4>

  ```python
  # Supervised Estimators
  y_pred = svc.predict(np.random.random(2,5))
  y_pred = lr.predict(X_test)
  y_pred = knn.predict_proba(X_test)
  # Unsupervised Estimators
  y_pred = k_means.predict(X_test)
  ```

<h4 id="8">8.模型性能评估</h4>

  ```python
  # 分类指标度量 ClassFication Metrics
  # 精度分数 Accuracy Score
  knn.score(X_test,y_test)
  from sklearn.metrics import accuracy_score
  accuracy_score(y_test,y_pred)
  # 分类报告 Classfication Report
  from sklearn.metrics import classification_report
  print(classification_report(y_test,y_pred))
  # 混淆矩阵 Confusion Matrix
  from sklearn.metrics import confusion_matrix
  print(confusion_matrix(y_test,y_pred))
  # 回归度量指标 Regression Metrics
  # 平均绝对误差 Mean Absolute Error
  from sklearn.metrics import mean_absolute_error
  y_true = [3,-0.5,2]
  mean_absolute_error(y_test,y_pred)
  # 均方差 Mean Squared Error
  from sklearn.metrics import mean_squared_error
  mean_squared_error(y_test,y_pred)
  # R平方分数 R^2 Score
  from sklearn.metrics import r2_score
  r2_score(y_true,y_pred)
  # 聚类指标 Clustering Metrics
  # 调整兰德指数ARI Adjust RandIndex
  from sklearn.metrics import adjust_rand_score
  adjust_rand_score(y_true,y_pred)
  # 异构性 Homogeneity
  from sklearn.metrics import homogeneity_score
  homogeneity_score(y_true,y_pred)
  # V-measure
  from sklearn.metrics import v_measure_score
  v_measure_score(y_true,y_pred)
  # 交叉验证 Cross-Validation
  from sklearn.cross_validation import cross_val_score
  print(cross_val_score(knn,X_train,y_train,cv=4))
  print(cross_val_score(lr,X,y,cv=2))
  ```

<h4 id="9">9.模型调整</h4>

  ```python
  # 超参数优化 Grid Search
  from sklearn.grid_search import GridSearchCV
  params = {"n_neighbors":np.arange(1,3),"metrics":["euclidean","cityblock"],}
  grid = GridSearchCV(estimator=knn,param_grid=params)
  grid.fit(X_tarin,y_train)
  print(grid.best_score_)
  print(grid.best_estimator_.n_neighbors)
  # 随机搜索参数 Randomized Parameter Optimization
  from sklearn.grid_search import RandomizedSearchCV
  params = {"n_neighbors":range(1,5),"weights":["uniform","distance"]}
  research = RandomizedSearchCV(estimator=knn,param_distributions=params,cv=4,n_iter=8,random_state=5)
  research.fit(X_train,y_train)
  print(rsearch.best_score_)
  ```

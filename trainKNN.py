
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 读取数据集的第一个块
chunk_size = 1000  # 每次读取的行数
data_chunks = pd.read_csv('D:\\train.csv', chunksize=chunk_size)
first_chunk = next(data_chunks)
first_chunk.fillna(first_chunk.mean(), inplace=True)

# 分割特征和目标变量
X = first_chunk.drop('target', axis=1)
y = first_chunk['target']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# 创建标准化器
scaler = StandardScaler()

# 对第一分块训练集去调用fit方法对标准化器进行适配训练。
scaler.fit(X_train)

# 标准化测试集
X_test = scaler.transform(X_test)

# 初始化 KNN 模型
knn = KNeighborsClassifier(n_neighbors=15)
# 逐块读取和训练
for chunk in data_chunks:

    # 分割特征和目标变量
    chunk.fillna(chunk.mean(), inplace=True)
    X = chunk.drop('target', axis=1)
    y = chunk['target']

    # 标准化每一分块的特征训练集
    X = scaler.fit_transform(X)

    # 拟合 KNN 模型
    knn.fit(X, y)


# 在测试集上进行预测
y_pred = knn.predict(X_test)

# 在测试集上评估模型性能
accuracy = round(knn.score(X_test, y_test), 5)
print("Accuracy on testing set: {:.3f}".format(roc_auc_score(y_test, y_pred)))
# 打印模型准确度
print("Accuracy:", accuracy)
print("classification report:\n", classification_report(y_test, knn.predict(X_test),
                                                        target_names=["非5g", "5g"]))

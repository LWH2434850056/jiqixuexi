import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# 加载CIFAR-10数据集
def load_cifar10_batch(batch_filename):
    cifar10_folder = 'cifar-10'
    full_path = os.path.join(cifar10_folder, batch_filename)
    with open(full_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    return data, labels


# 加载所有数据
def load_cifar10_data():
    data, labels = [], []
    for i in range(1, 6):
        batch_filename = f"data_batch_{i}"
        batch_data, batch_labels = load_cifar10_batch(batch_filename)
        data.append(batch_data)
        labels.append(batch_labels)
    test_batch_filename = "test_batch"
    test_data, test_labels = load_cifar10_batch(test_batch_filename)

    # 将数据合并成一个大数组
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels, test_data, test_labels


# 加载数据集
train_data, train_labels, test_data, test_labels = load_cifar10_data()

# 数据标准化
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# 标签one-hot编码
lb = LabelBinarizer()
train_labels_one_hot = lb.fit_transform(train_labels)
test_labels_one_hot = lb.transform(test_labels)

# 降维 (PCA) 用于加速训练, 设定维度为50以减少计算量
pca = PCA(n_components=50)
train_data_pca = pca.fit_transform(train_data.reshape(-1, 32 * 32 * 3))
test_data_pca = pca.transform(test_data.reshape(-1, 32 * 32 * 3))

# 训练SVM
svm = SVC(kernel='linear', C=1, probability=True, verbose=False)
svm.fit(train_data_pca, train_labels)

# 预测和评估 SVM
svm_preds = svm.predict(test_data_pca)
print("SVM Classification Report:")
print(classification_report(test_labels, svm_preds))

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=0)  # 使用并行化
rf.fit(train_data_pca, train_labels)

# 预测和评估 随机森林
rf_preds = rf.predict(test_data_pca)
print("Random Forest Classification Report:")
print(classification_report(test_labels, rf_preds))

# 训练KNN
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  # 使用并行化
knn.fit(train_data_pca, train_labels)

# 预测和评估 KNN
knn_preds = knn.predict(test_data_pca)
print("KNN Classification Report:")
print(classification_report(test_labels, knn_preds))


# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


labels = [str(i) for i in range(10)]
cm = confusion_matrix(test_labels, rf_preds)
plot_confusion_matrix(cm, labels)

# 训练KMeans进行聚类 (无监督学习)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(train_data_pca)

# 聚类预测
kmeans_preds = kmeans.predict(test_data_pca)

# 无监督评估
ari_score = adjusted_rand_score(test_labels, kmeans_preds)
print(f"Adjusted Rand Index for KMeans Clustering: {ari_score:.4f}")


# 可视化分类模型训练准确率曲线
def plot_accuracy_curve(model, train_data, train_labels, test_data, test_labels, model_name):
    model.fit(train_data, train_labels)  # 使用全部训练数据
    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)

    train_accuracy = accuracy_score(train_labels, train_preds)
    test_accuracy = accuracy_score(test_labels, test_preds)

    print(f'{model_name} Training Accuracy: {train_accuracy:.4f}')
    print(f'{model_name} Test Accuracy: {test_accuracy:.4f}')
# 绘制训练和测试准确率曲线
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2], [train_accuracy, test_accuracy], marker='o', label=f'{model_name} Accuracy')
    plt.xticks([1, 2], ['Train', 'Test'])
    plt.ylim([0, 1])
    plt.title(f'{model_name} Accuracy Curve')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# SVM训练曲线
plot_accuracy_curve(svm, train_data_pca, train_labels, test_data_pca, test_labels, 'SVM')

# 随机森林训练曲线
plot_accuracy_curve(rf, train_data_pca, train_labels, test_data_pca, test_labels, 'Random Forest')

# KNN训练曲线
plot_accuracy_curve(knn, train_data_pca, train_labels, test_data_pca, test_labels, 'KNN')

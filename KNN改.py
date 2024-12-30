import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


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

# 训练KNN
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  # 使用并行化
knn.fit(train_data_pca, train_labels)


# 预测和评估 KNN
knn_preds = knn.predict(test_data_pca)
print("KNN Classification Report:")
print(classification_report(test_labels, knn_preds))

# 可视化模型训练准确率变化
def plot_accuracy_curve_knn(train_data, train_labels, test_data, test_labels, model, model_name):
    train_accuracies = []
    test_accuracies = []

    # 每次增加数据的大小
    for i in range(10, len(train_data), 1000):  # 每次增加1000个训练样本
        model.fit(train_data[:i], train_labels[:i])  # 使用增量大小的训练集

        # 计算训练集和测试集的准确率
        train_preds = model.predict(train_data[:i])
        test_preds = model.predict(test_data)

        train_accuracy = accuracy_score(train_labels[:i], train_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    print(f'{model_name} Training Accuracy: {train_accuracy:.4f}')
    print(f'{model_name} Test Accuracy: {test_accuracy:.4f}')

    # 绘制训练准确率和测试准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, len(train_data), 1000), train_accuracies, label='Train Accuracy')
    plt.plot(range(10, len(train_data), 1000), test_accuracies, label='Test Accuracy', linestyle='--')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


# 可视化混淆矩阵
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=lb.classes_, yticklabels=lb.classes_)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# 训练KNN并可视化混淆矩阵
knn.fit(train_data_pca, train_labels)
test_preds = knn.predict(test_data_pca)

# 绘制训练准确率和测试准确率曲线
plot_accuracy_curve_knn(train_data_pca, train_labels, test_data_pca, test_labels, knn, 'KNN')

# 绘制混淆矩阵
plot_confusion_matrix(test_labels, test_preds, 'KNN')

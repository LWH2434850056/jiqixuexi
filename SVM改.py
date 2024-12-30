import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
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

# 降维 (PCA) 用于加速训练 - 降维维度减小到50
pca = PCA(n_components=50)  # 降维到50，减少计算量
train_data_pca = pca.fit_transform(train_data.reshape(-1, 32 * 32 * 3))
test_data_pca = pca.transform(test_data.reshape(-1, 32 * 32 * 3))

# 随机抽取部分训练数据以减少时间
np.random.seed(42)  # 设置随机种子以保证可复现性
indices = np.random.choice(len(train_data_pca), size=5000, replace=False)  # 抽取5000个样本
train_data_pca = train_data_pca[indices]
train_labels = train_labels[indices]

# 训练SVM
svm = SVC(kernel='linear', C=1, probability=True)  # 移除了n_jobs参数
svm.fit(train_data_pca, train_labels)


# 预测和评估 SVM
svm_preds = svm.predict(test_data_pca)
print("SVM Classification Report:")
print(classification_report(test_labels, svm_preds))


# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


labels = [str(i) for i in range(10)]
cm = confusion_matrix(test_labels, svm_preds)
plot_confusion_matrix(cm, labels)


# 可视化SVM分类器训练和测试准确率曲线
def plot_accuracy_curve(model, train_data, train_labels, test_data, test_labels, model_name):
    # 只使用部分样本评估，减少时间
    step_size = 1000
    train_accuracies = []
    test_accuracies = []

    for i in range(step_size, len(train_data), step_size):  # 每1000步评估一次
        model.fit(train_data[:i], train_labels[:i])
        train_preds = model.predict(train_data[:i])
        test_preds = model.predict(test_data)
        train_accuracies.append(accuracy_score(train_labels[:i], train_preds))
        test_accuracies.append(accuracy_score(test_labels, test_preds))

    plt.plot(range(step_size, len(train_data), step_size), train_accuracies, label=f'{model_name} Train')
    plt.plot(range(step_size, len(train_data), step_size), test_accuracies, label=f'{model_name} Test')
    plt.xlabel('Training Samples')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Training and Testing Accuracy')
    plt.legend()
    plt.show()


# SVM训练曲线
plot_accuracy_curve(svm, train_data_pca, train_labels, test_data_pca, test_labels, 'SVM')

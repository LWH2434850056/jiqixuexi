import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# 降维 (PCA) 用于加速计算，设定维度为50以减少计算量
pca = PCA(n_components=50)
train_data_pca = pca.fit_transform(train_data.reshape(-1, 32 * 32 * 3))
test_data_pca = pca.transform(test_data.reshape(-1, 32 * 32 * 3))

# 训练KMeans进行聚类 (无监督学习)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(train_data_pca)

# 聚类预测
kmeans_preds = kmeans.predict(test_data_pca)

# 无监督评估
ari_score = adjusted_rand_score(test_labels, kmeans_preds)
print(f"Adjusted Rand Index for KMeans Clustering: {ari_score:.4f}")

# 分类报告
report = classification_report(test_labels, kmeans_preds, output_dict=True)
print("Classification Report:")
print(report)

# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

labels = [str(i) for i in range(10)]
cm = confusion_matrix(test_labels, kmeans_preds)
plot_confusion_matrix(cm, labels)

# 模拟模型训练过程，绘制准确率和损失的变化曲线
# 这里我们只是模拟数据，实际应用中应该有真实的训练过程
epochs = 10
train_acc = np.random.rand(epochs)  # 随机生成训练准确率
val_acc = np.random.rand(epochs)  # 随机生成验证准确率
train_loss = np.random.rand(epochs)  # 随机生成训练损失
val_loss = np.random.rand(epochs)  # 随机生成验证损失

# 绘制训练和验证准确率的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_acc, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_acc, label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 绘制训练和验证损失的变化曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

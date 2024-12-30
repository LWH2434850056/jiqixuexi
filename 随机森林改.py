import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

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

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=0)  # 使用并行化
rf.fit(train_data.reshape(-1, 32 * 32 * 3), train_labels)

# 预测和评估 随机森林
rf_preds = rf.predict(test_data.reshape(-1, 32 * 32 * 3))
print("Random Forest Classification Report:")
print(classification_report(test_labels, rf_preds))

# 可视化混淆矩阵
def plot_confusion_matrix(cm, labels, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 正则化
        print("Normalized confusion matrix")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

labels = [str(i) for i in range(10)]
cm = confusion_matrix(test_labels, rf_preds)
plot_confusion_matrix(cm, labels, normalize=True)

# 可视化随机森林训练准确率曲线
def plot_accuracy_curve(model, train_data, train_labels, test_data, test_labels, model_name):
    train_accuracies = []
    test_accuracies = []

    # 使用交叉验证来估计模型在不同训练阶段的准确率
    # 这里我们仅采用KFold交叉验证来观察不同划分下的模型表现
    for i in range(1, 101):
        model.set_params(n_estimators=i)  # 动态修改树的数量
        model.fit(train_data, train_labels)  # 训练
        train_preds = model.predict(train_data)
        test_preds = model.predict(test_data)

        train_accuracy = accuracy_score(train_labels, train_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    print(f'{model_name} Training Accuracy: {train_accuracy:.4f}')
    print(f'{model_name} Test Accuracy: {test_accuracy:.4f}')

    # 绘制准确率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), train_accuracies, label="Training Accuracy", color='b')
    plt.plot(range(1, 101), test_accuracies, label="Test Accuracy", color='r')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# 随机森林训练曲线
plot_accuracy_curve(rf, train_data.reshape(-1, 32 * 32 * 3), train_labels, test_data.reshape(-1, 32 * 32 * 3), test_labels, 'Random Forest')
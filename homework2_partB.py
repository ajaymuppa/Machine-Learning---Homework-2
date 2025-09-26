import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

# - loads Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Q7 Decision Trees
depths = [1,2,3]
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print((d, train_acc, test_acc))

# Q8 kNN decision boundaries (first two features)
X2 = iris.data[:, :2]
y2 = iris.target
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42, stratify=y2)

scaler = StandardScaler().fit(X2_train)
X2_train_s = scaler.transform(X2_train)
X2_test_s = scaler.transform(X2_test)

def plot_decision_boundary(X, y, clf, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.xlabel('sepal length (standardized)')
    plt.ylabel('sepal width (standardized)')
    plt.title(title)
    plt.show()

ks = [1,3,5,10]
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X2_train_s, y2_train)
    train_acc = accuracy_score(y2_train, knn.predict(X2_train_s))
    test_acc = accuracy_score(y2_test, knn.predict(X2_test_s))
    print((k, train_acc, test_acc))
    plot_decision_boundary(np.vstack((X2_train_s, X2_test_s)), np.hstack((y2_train, y2_test)), knn, f"k-NN decision boundary (k={k})")

# Q9 kNN (k=5) full features, confusion matrix, classification report, ROC + AUC
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
y_pred = knn5.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
print("Classification report:\n", classification_report(y_test, y_pred, target_names=target_names, digits=4))

y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_score = knn5.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

fpr = dict(); tpr = dict(); roc_auc = dict()
plt.figure(figsize=(6,5))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"class {i} (AUC = {roc_auc[i]:.3f})")

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
roc_auc["macro"] = auc(all_fpr, mean_tpr)

plt.plot(fpr["micro"], tpr["micro"], linestyle=':', label=f"micro (AUC = {roc_auc['micro']:.3f})")
plt.plot(all_fpr, mean_tpr, linestyle='--', label=f"macro (AUC = {roc_auc['macro']:.3f})")
plt.plot([0,1],[0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves (one-vs-rest)')
plt.legend()
plt.show()

print("AUC:", roc_auc)

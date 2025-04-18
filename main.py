import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("processed_titanic.csv")
print(df.head())
print(df.dtypes)
df.drop(columns=["Name"], inplace=True)

X = df.drop(columns=["Survived", "Age"])
ycls = df["Survived"]
yr = df["Age"]

X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X, ycls, test_size=0.4, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, yr, test_size=0.4, random_state=42)

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

reg = DecisionTreeRegressor(max_depth=5, random_state=42)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plot_tree(reg, feature_names=X.columns, filled=True)
plt.title("Регрессионное дерево (предсказание возраста)")
plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
clf = DecisionTreeClassifier( random_state=42)
clf.fit(X_train_cl, y_train_cl)
y_pred_class = clf.predict(X_test_cl)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_cl, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Не выжил", "Выжил"], yticklabels=["Не выжил", "Выжил"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=X.columns, class_names=["Не выжил", "Выжил"], filled=True)
plt.title("Классификационное дерево (предсказание выживания)")
plt.show()

y_proba = clf.predict_proba(X_test_cl)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_cl, y_proba)
roc_auc = auc(fpr, tpr)

print("ROC-AUC:", roc_auc)

plt.plot(fpr, tpr, marker='o')
plt.plot([0, 1.1], [0, 1.1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC-кривая")
plt.grid()
plt.show()

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, roc_auc_score)
from sklearn.utils.validation import check_is_fitted

# Load data
d = pd.read_csv('creditcardsampling.csv')
df = pd.read_csv('creditcardsampling.csv')

# Visualizations
sns.histplot(d['Amount'], kde=True)  # Add KDE for better visualization
sns.histplot(d['Time'], kde=True)
d.hist(figsize=(20, 20))
plt.show()
sns.jointplot(x='Time', y='Amount', data=d)

# Data Sampling
class0 = d[d['Class'] == 0]
class1 = d[d['Class'] == 1]
temp = shuffle(class0)
d1 = temp.iloc[:2000, :]
frames = [d1, class1]
df_temp = pd.concat(frames)
df = shuffle(df_temp)
df.to_csv('creditcardsampling.csv', index=False)  # Ensure index is not saved

# Visualization of Class Distribution
sns.countplot(x='Class', data=df)
plt.show()

# SMOTE
oversample = SMOTE()
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X, Y = oversample.fit_resample(X, Y)

# Convert to DataFrame
X = pd.DataFrame(X, columns=df.columns[:-1])
Y = pd.DataFrame(Y, columns=['Class'])
data = pd.concat([X, Y], axis=1)

# Feature Scaling
scaler = StandardScaler()
frames = ['Time', 'Amount']
x = data[frames]
d_temp = data.drop(frames, axis=1)
scaled_col = pd.DataFrame(scaler.fit_transform(x), columns=frames)
d_scaled = pd.concat([scaled_col, d_temp], axis=1)

# Dimensionality Reduction
pca = PCA(n_components=7)
X_temp_reduced = pca.fit_transform(d_scaled)
names = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
X_reduced = pd.DataFrame(X_temp_reduced, columns=names)
new_data = pd.concat([X_reduced, d_scaled['Class']], axis=1)
new_data.to_csv('finaldata.csv', index=False)  # Ensure index is not saved

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, d_scaled['Class'], test_size=0.3, random_state=42)


# Logistic Regression
lr = LogisticRegression(max_iter=1000)  # Set max_iter if needed
lr = LogisticRegression(solver='liblinear', penalty='l1', max_iter=1000)  # Increase max_iter if needed

try:
    lr.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")

try:
    y_pred_lr = lr.predict(X_test)
    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
except Exception as e:
    print(f"Error during prediction: {e}")


# Hyperparameter Tuning for Logistic Regression
r_params = {'penalty': ['l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['liblinear'], 'max_iter': [1000]}
grid_lr = GridSearchCV(lr, param_grid= r_params)

try:
    grid_lr.fit(X_train, y_train)
    print("Best Params:", grid_lr.best_params_)
    y_pred_lr3 = grid_lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr3))
except Exception as e:
    print(f"Error during grid search: {e}")

    
try:
    check_is_fitted(lr)
    y_pred_lr = lr.predict(X_test)
except Exception as e:
    print(f"Model not fitted: {e}")


# Support Vector Machine
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print(classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))

# Hyperparameter Tuning for SVC
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 1, 0.01, 0.0001, 0.001]}]
grid_search = GridSearchCV(estimator=svc, param_grid=parameters, scoring='accuracy', n_jobs=-1, cv=5)
grid_search.fit(X_train, y_train)
print("Best Accuracy: {:.2f} %".format(grid_search.best_score_ * 100))
print("Best Parameters:", grid_search.best_params_)
svc_param = SVC(kernel='rbf', gamma=0.01, C=100)
svc_param.fit(X_train, y_train)
y_pred_svc2 = svc_param.predict(X_test)
print(classification_report(y_test, y_pred_svc2))

# Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
print(classification_report(y_test, y_pred_dtree))
print(confusion_matrix(y_test, y_pred_dtree))

# Hyperparameter Tuning for Decision Tree
tree_parameters = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(2, 4)), 'min_samples_leaf': list(range(5, 7))}
grid_tree = GridSearchCV(dtree, tree_parameters, cv=5)
grid_tree.fit(X_train, y_train)
y_pred_dtree2 = grid_tree.predict(X_test)
print(classification_report(y_test, y_pred_dtree2))






# Random Forest
randomforest = RandomForestClassifier(n_estimators=5)
randomforest.fit(X_train, y_train)
y_pred_rf = randomforest.predict(X_test)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))

# Hyperparameter Tuning for KNN
knn_params = {"n_neighbors": list(range(2, 5)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, cv=5)
grid_knn.fit(X_train, y_train)
print("Best Params:", grid_knn.best_params_)
knn = KNeighborsClassifier(n_neighbors=grid_knn.best_params_['n_neighbors'])
knn.fit(X_train, y_train)
pred_knn2 = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn2))
print(classification_report(y_test, pred_knn2))

# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xg = xgb.predict(X_test)
print(classification_report(y_test, y_pred_xg))

# LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
parameters = {'num_leaves': 2**8, 'learning_rate': 0.1, 'is_unbalance': True, 'objective': 'binary'}
num_rounds = 300
clf = lgb.train(parameters, lgb_train, num_boost_round=num_rounds)
y_prob = clf.predict(X_test)
y_pred = np.where(y_prob > 0.5, 1, 0)
print(classification_report(y_test, y_pred))

# ROC Curve
lg_fpr, lg_tpr, _ = roc_curve(y_test, y_pred_lr3)
svc_fpr, svc_tpr, _ = roc_curve(y_test, y_pred_svc2)
dtree_fpr, dtree_tpr, _ = roc_curve(y_test, y_pred_dtree2)
rf_fpr, rf_tpr, _ = roc_curve(y_test, y_pred_rf)
knn_fpr, knn_tpr, _ = roc_curve(y_test, pred_knn2)
xg_fpr, xg_tpr, _ = roc_curve(y_test, y_pred_xg)
lgb_fpr, lgb_tpr, _ = roc_curve(y_test, y_pred)

plt.figure(figsize=(15, 10))
plt.title("ROC Curve")
plt.plot(lg_fpr, lg_tpr, label='Logistic Regression Classifier: {:.4f}'.format(roc_auc_score(y_test, y_pred_lr3)))
plt.plot(knn_fpr, knn_tpr, label='KNN Classifier: {:.4f}'.format(roc_auc_score(y_test, pred_knn2)))
plt.plot(svc_fpr, svc_tpr, label='SVC Classifier: {:.4f}'.format(roc_auc_score(y_test, y_pred_svc2)))
plt.plot(dtree_fpr, dtree_tpr, label='Decision Tree Classifier: {:.4f}'.format(roc_auc_score(y_test, y_pred_dtree2)))
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier: {:.4f}'.format(roc_auc_score(y_test, y_pred_rf)))
plt.plot(xg_fpr, xg_tpr, label='XGBoost Classifier: {:.4f}'.format(roc_auc_score(y_test, y_pred_xg)))
plt.plot(lgb_fpr, lgb_tpr, label='LightGBM Classifier: {:.4f}'.format(roc_auc_score(y_test, y_pred)))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend()
plt.show()

from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
import pickle

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

training_file = '/ETH-scam-ml/modded/eth_txn_unscaled_Vars.csv' # aggregated data

# read the data into a pandas dataframe and split into X and y and then split into train and test sets
df = pd.read_csv(training_file)
X = df.drop(columns=['address', 'flag'])
y = df['flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Log for Skewed Data
columns = list(X.columns)
for c in columns:
    X_train[c] = X_train[c].apply(lambda x: np.log(x) if x > 0 else 0)
    X_test[c]  = X_test[c].apply(lambda x: np.log(x) if x > 0 else 0)

# Scaling
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# output scalar to pickle

pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Convert to DataFrames
X_train_df = pd.DataFrame(X_train, columns=columns)
X_test_df  = pd.DataFrame(X_test, columns=columns)

# Grid Search for Hyperparameters
params = {'max_depth': range(2, 10, 1),
          'n_estimators': range(1000, 1500, 100),
          'learning_rate': [0.1, 0.01, 0.05]}
# XGBoost Classifier
clf = XGBClassifier(objective='binary:logistic',
                    seed=42, n_jobs=-1, verbosity=0, silent=True)

tuned_clf = GridSearchCV(estimator=clf,
                         param_grid=params,
                         scoring='f1',
                         cv=5,
                         verbose=0,).fit(X_train,y_train)

print("Tuned Hyperparameters :", tuned_clf.best_params_)
print("Train F1 Score :",tuned_clf.best_score_)
best_y_pr = tuned_clf.predict(X_test)
print('Test F1 Score: ', f1_score(y_test, best_y_pr))

from matplotlib import pyplot as plt
feat_importances = tuned_clf.best_estimator_.feature_importances_
indices = np.argsort(feat_importances)
# plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.title("Ranked feature importances")
plt.barh(range(len(feat_importances)), feat_importances[indices], align="center")
features = ['feature_{}'.format(columns[i]) for i in range(len(columns))]
plt.yticks(range(len(feat_importances)), [features[idx] for idx in indices])
plt.ylim([-1, len(feat_importances)])
plt.show();

def model_diagnostic_stats(confusion_matrix):
    """Returns a dictionary of model diagnostic statistics."""
    tp = confusion_matrix[1,1]
    tn = confusion_matrix[0,0]
    fp = confusion_matrix[0,1]
    fn = confusion_matrix[1,0]

    p  = tp + fn
    n  = fp + tn
    pp = tp + fp
    pn = fn + tn

    diagnostic_dict = {'recall' : tp/p,
                       'false_neg_rate' : fn/p,
                       'false_pos_rate' : fp/n,
                       'true_neg_rate' : tn/n,
                       'positive_liklihood_ratio' : (tp/p)/(fp/n),
                       'neg_liklihood_rate' : (fn/p)/(tn/n),
                       'precision' : tp/pp,
                       'false_omission_rate' : fn/pn,
                       'false_discovery_rate' : fp/pp,
                       'neg_pred_value' : tn/pn,
                       'markedness' : (tp/pp)+(tn/pn)-1,
                       'diagnostic_odds_ration' : ((tp/p)/(fp/n))/((fn/p)/(tn/n)),
                       'informedness' : (tp/p)+(tn/n)-1,
                       'prevalence_threshold' : (sqrt((tp/p)*(fp/n))-(fp/n))/((tp/p)-(fp/n)),
                       'prevalence' : p/(p+n),
                       'accuracy' : (tp+tn)/(p+n),
                       'balanced_accuracy' : ((tp/p)+(tn/n))/2,
                       'F1_score' : 2*tp/(2*tp+fp+fn),
                       'fowlkes_mallows_index' : sqrt((tp/pp)*(tp/p)),
                       'jaccard_index' : tp/(tp+fn+fp),}

    return diagnostic_dict

#tuned_clf.best_estimator_
#tuned_clf.best_score_
#tuned_clf.best_params_

# Predict on test set
y_pred = tuned_clf.predict(X_test)
y_pred_prob = tuned_clf.predict_proba(X_test)[:,1]

xgb_accuracy_score = accuracy_score(y_test, y_pred)
xgb_auc_score = roc_auc_score(y_test, y_pred)
print('XGBoost model accuracy score: {0:0.4f} '
      'and roc_auc score: {0:0.4f}'. format(xgb_accuracy_score,xgb_auc_score))

cm_matrix = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(5,5))
sb.heatmap(cm_matrix,annot=True,fmt='g',cmap="Blues")
stats = model_diagnostic_stats(cm_matrix)
print('Model Confusion Matrix Statistics:')
for key,value in stats.items():
    value_str = '%.4f' % value
    print("\n {}: {}".format(key,value_str))



feature_importances = tuned_clf.best_estimator_.feature_importances_
column_var = X_train_df.columns.tolist()
feature_imprt_df = pd.DataFrame({'Features':column_var, '% importance': feature_importances})
formatted_vars_display = [ '%.3f' % elem for elem in feature_importances ]
ax = sb.barplot(x='% importance', y='Features',data=feature_imprt_df)
ax.set(xlabel='% importance', ylabel='Features')
plt.show()

# Save the model
filename = 'xgboost_model.pkl'
pickle.dump(tuned_clf, open(filename, 'wb'))
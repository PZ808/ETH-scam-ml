import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

with open('xgboost_model.pkl', 'rb') as f:
    xgb_cl = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# read the data from a pandas dataframe and split into X and y and then split into train and test sets
# read the data from google cloud storage
gs_location = 'test_data.csv'
df = pd.read_csv(gs_location, index_col=False)


# split into X and y
X = df.drop(columns=['address'])

columns = list(X.columns)
for c in columns:
    X[c] = X[c].apply(lambda x: np.log(x) if x > 0 else 0)

X = scaler.transform(X)
X_df = pd.DataFrame(X, columns=columns)

y_pred = xgb_cl.predict(X_df)
y_pred_proba = xgb_cl.predict_proba(X_df)


if y_pred[0] == 0:
    print('The model predicts that the address {:s} is clean with probability {:.2f}'.format(str(df.address.iloc[0]),
                                                                                             y_pred_proba[0][0]))
else:
    print('The model predicts that the address {:s} is dirty with probability {:.2f}'.format(str(df.address.iloc[0]),
                                                                                             y_pred_proba[0][1]))




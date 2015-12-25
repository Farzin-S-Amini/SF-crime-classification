from __future__ import print_function, division
import numpy as np

import pandas as pd

from patsy import dmatrices, dmatrix
from sklearn.ensemble import RandomForestClassifier

def main():
    # Read in the training and testing data
    df_train = pd.read_csv('train.csv', parse_dates=['Dates'])
    df_train.drop(['Descript', 'Dates', 'Resolution',
                   'Address'], axis=1, inplace=True)
    df_test = pd.read_csv('test.csv', parse_dates=['Dates'])
    df_test.drop(['Dates', 'Address'], axis=1, inplace=True)

    # Select training and validation sets
    inds = np.arange(df_train.shape[0])
    np.random.shuffle(inds)
    train_inds = inds[:int(0.2 * df_train.shape[0])]
    val_inds = inds[int(0.2 * df_train.shape[0]):]

    # Extract the column names
    col_names = np.sort(df_train['Category'].unique())

    # Recode categories to numerical
    df_train['Category'] = pd.Categorical.from_array(df_train['Category']).codes
    df_train['DayOfWeek'] = pd.Categorical.from_array(df_train['DayOfWeek']).codes
    df_train['PdDistrict'] = pd.Categorical.from_array(df_train['PdDistrict']).codes
    df_test['DayOfWeek'] = pd.Categorical.from_array(df_test['DayOfWeek']).codes
    df_test['PdDistrict'] = pd.Categorical.from_array(df_test['PdDistrict']).codes

    # Split up the training and validation sets
    df_val = df_train.ix[val_inds]
    df_train = df_train.ix[train_inds]

    # Construct the design matrix and response vector for the
    # training data and the design matrix for the test data
    y_train, X_train = dmatrices('Category ~ X + Y + DayOfWeek + PdDistrict', df_train)
    y_val, X_val = dmatrices('Category ~ X + Y + DayOfWeek + PdDistrict', df_val)
    X_test = dmatrix('X + Y + DayOfWeek + PdDistrict', df_test)

    # Fit the random forest
    randforest = RandomForestClassifier(n_estimators=11)
    randforest.fit(X_train, y_train.ravel())
    print('Mean accuracy (Random Forest): {:.4f}'.format(randforest.score(X_val, y_val.ravel())))
    # Make predictions
    predict_probs = randforest.predict_proba(X_test)

    print(predict_probs.shape, col_names.shape)

    # Add the id numbers for the incidents and construct the final df
    df_pred = pd.DataFrame(data=predict_probs, columns=col_names)
    df_pred['Id'] = df_test['Id'].astype(int)
    df_pred.to_csv('output.csv', index=False)

if __name__ == '__main__':
    main()
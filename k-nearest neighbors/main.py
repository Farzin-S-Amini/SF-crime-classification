import pandas as pd
import numpy as np
import zipfile
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train.csv', parse_dates=['Dates'])[['X', 'Y', 'Category']]


# Separate test and train set out of orignal train set.
msk = np.random.rand(len(train)) < 0.8
knn_train = train[msk]
knn_test = train[~msk]


# Prepare data sets
x = knn_train[['X', 'Y']]
y = knn_train['Category'].astype('category')
actual = knn_test['Category'].astype('category')
z = knn_test[['X', 'Y']]

# Fit
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(x, y)
score = knn.score(z,actual)
print('Mean accuracy: {:.4f}'.format(score))



# Submit for K=40
z = zipfile.ZipFile('../input/test.csv.zip')
test = pd.read_csv(z.open('test.csv'), parse_dates=['Dates'])
x_test = test[['X', 'Y']]
knn = KNeighborsClassifier(n_neighbors=40)
knn.fit(x, y)
outcomes = knn.predict(x_test)

submit = pd.DataFrame({'Id': test.Id.tolist()})
for category in y.cat.categories:
    submit[category] = np.where(outcomes == category, 1, 0)
    
submit.to_csv('k_nearest_neigbour.csv', index = False)

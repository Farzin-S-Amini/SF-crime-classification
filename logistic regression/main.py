__author__ = 'farzin'

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


#Load Data with pandas, and parse the first column into datetime
train=pd.read_csv('train.csv', parse_dates = ['Dates'])
#Convert crime labels to numbers
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Category)

#Get binarized weekdays, districts, and hours.
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour)

#Build new array
train_data = pd.concat([hour, days, district], axis=1)
train_data['crime']=crime


features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

features2 = [x for x in range(0,24)]
features = features + features2

training, validation = train_test_split(train_data, train_size=.60)

#Logistic Regression for comparison
model = LogisticRegression(C=1)
model.fit(training[features], training['crime'])
score = model.score(validation[features],validation['crime'])
print(score)
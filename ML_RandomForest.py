import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df=pd.read_csv(r"car_evaluation.csv")

X = df.drop(['unacc'], axis=1)
y = df['unacc']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

import category_encoders as ce
# encode categorical variables with ordinal encoding
encoder = ce.OrdinalEncoder(cols=['vhigh', 'vhigh.1', '2', '2.1', 'small', 'low'])


X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

clf = RandomForestClassifier(n_estimators=100, random_state=0)

clf.fit(X_train, y_train)
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_scores)

sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
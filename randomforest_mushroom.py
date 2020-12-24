import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
import seaborn as sns

data = pd.read_csv("mushrooms.csv")

sns.countplot(data["class"])

le = LabelEncoder()
for i in data.columns:
    data[i] = le.fit_transform(data[i])

sns.heatmap(data.corr())

X = data.drop(["class","stalk-surface-below-ring","stalk-color-below-ring","gill-attachment","gill-color","cap-shape", "stalk-shape"], axis = 1)
y = np.array(data["class"])

rfc = RandomForestClassifier()
score = cross_val_score(rfc, X, y, cv=15)
pred = cross_val_predict(rfc, X, y, cv=15)
print((sum(score)/15)*100)

model = sm.OLS(pred, X).fit()
print(model.summary())

cm = confusion_matrix(y, pred)
print(cm)
sns.heatmap(cm, annot = True, cmap = "rocket_r", fmt='g')

print(classification_report(y, pred))

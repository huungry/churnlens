import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

y_test = joblib.load('model/y_test.joblib')
y_pred = joblib.load('model/y_pred.joblib')

cm = confusion_matrix(y_test, y_pred)
print(cm)

dis = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Actually No Churn (0)', 'Actually Churn (1)'])

fig, ax = plt.subplots(figsize=(8, 5)) 
dis.plot(cmap=plt.cm.Blues, ax=ax)
plt.savefig('assets/confusion_matrix.png')
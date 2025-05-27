import  pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from  sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
data=pd.read_csv("SMSSpamCollection", sep="\t")
data.columns=["tipo", "mensaje"]
vectorizer= TfidfVectorizer()
vectorMensaje= vectorizer.fit_transform(data["mensaje"])
data["tipo"]= data["tipo"].map({"ham": 1, "spam": 2})
x_train, x_test, y_train, y_test= train_test_split(vectorMensaje, data["tipo"], test_size=0.2, random_state=42)
model= MultinomialNB()
model.fit(x_train, y_train)
y_pred= model.predict(x_test)
evaluacion=accuracy_score(y_test, y_pred)
cm=confusion_matrix(y_test, y_pred)
disp= ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=["Ham", "Spam"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusion")
print(evaluacion)

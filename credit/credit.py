import shutil
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix

# arquivo= 'C:/Users/edubo/Desktop/I.A/pre-processamento/credito/credit.pkl'
# destino='C:/Users/edubo/Desktop/I.A/Aprendizagem baseado em instâncias/knn/credit'
# shutil.copy(arquivo,destino)

with open('credit.pkl','rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste = pickle.load(f)

knn_credit=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn_credit.fit(X_treinamento,y_treinamento)

previsoes=knn_credit.predict(X_teste)
accuracy_score(y_teste,previsoes) #98

cm=ConfusionMatrix(knn_credit)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)
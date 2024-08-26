import shutil
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

arquivo='C:/Users/edubo/Desktop/I.A/pre-processamento/censo/census.pkl'
destino='C:/Users/edubo/Desktop/I.A/Aprendizagem baseado em instâncias/knn/census'
shutil.copy(arquivo,destino)

with open('census.pkl','rb') as f:
    X_treinamento,y_treinamento,X_teste,y_teste= pickle.load(f)

knn_census=KNeighborsClassifier(n_neighbors=20,metric='minkowski',p=2)
knn_census.fit(X_treinamento,y_treinamento)

previsoes=knn_census.predict(X_teste)
accuracy_score(y_teste,previsoes) #83%

cm=ConfusionMatrix(knn_census)
cm.fit(X_treinamento,y_treinamento)
cm.score(X_teste,y_teste)
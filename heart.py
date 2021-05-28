import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, classification_report

# carregando dados
df = pd.read_csv("heart.csv")

# olhando alguns detalhes
print(df.head())
print(df.describe())

plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

df.info()
df.hist()

plt.figure(figsize=(28.5, 18))

#criar um df só pra arvore de decisao
cols_X = df[['age','cp','trestbps','restecg','thalach','exang','oldpeak','slope','ca','thal']]
cols_y = df[['target']]
df_X = cols_X.copy()
df_y = cols_y.copy()
print(df_X)
print(df_y)

#separando o ds em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(df_X, df_y, test_size=0.5, random_state=42)

#parâmetros básicos
clf_arvore = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=1)

#fase de treinamento
clf_arvore = clf_arvore.fit(X_treino, y_treino)
y_predicao_treino = clf_arvore.predict(X_treino)
print("Acuracidade no treino com DT:", accuracy_score(y_treino, y_predicao_treino))
print("Precisão no treino com DT:", precision_score(y_treino, y_predicao_treino, average='weighted'))

#matriz de confusão - para treino
print(classification_report(y_treino, y_predicao_treino))

#fase de testes
y_predicao_teste = clf_arvore.predict(X_teste)

#resultado nos testes
print("Acuracidade no teste com DT:", accuracy_score(y_teste, y_predicao_teste))
print("Precisão no teste com DT:", precision_score(y_teste, y_predicao_teste, average='weighted'))

tree.plot_tree(clf_arvore, fontsize=12)
plt.show()
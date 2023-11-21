import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data_bvsp = pd.read_csv('./dados/^BVSP.csv')
data_BRL = pd.read_csv('./dados/BRL=X.csv')
data_GOL = pd.read_csv('./dados/GOLL4.SA.csv')
data_ouro_tratado = pd.read_csv('./dados/Ouro_Tratado.csv')
data_petroleo_tratado = pd.read_csv('./dados/Petroleo_Tratado.csv')


# Organizando as datas dos arquivos de Ações Gol, Bovespa Ouro, Dólar e Petróleo

### Checa por valores nulos/etc nas colunas ###
#print(data_GOL.isnull().any())
#print(data_bvsp.isnull().any())
#print(data_ouro_tratado.isnull().any())
#print(data_BRL.isnull().any())
#print(data_petroleo_tratado.isnull().any())

#Dados GOL - organização e tratamento das Datas do arquivo
data_GOL['Date'] = pd.to_datetime(data_GOL['Date'])
data_GOL = data_GOL.sort_values('Date')
data_GOL.rename(columns={'Date': 'Data'}, inplace={True})

#Obter primeira e última entradas de data para definir intervalo de dados
primeira_data_GOL = data_GOL['Data'].loc[0]
ultima_data_GOL = data_GOL['Data'].iloc[-1]

#Dados Bovespa - organização e tratamento das Datas do arquivo
data_bvsp['Date'] = pd.to_datetime(data_bvsp['Date'])
data_bvsp = data_bvsp.sort_values('Date')
data_bvsp.rename(columns={'Date': 'Data'}, inplace={True})

#Ouro - Corrige, e tira das entradas as datas com valores inexistentes
data_ouro_tratado['Data'] = pd.to_datetime(data_ouro_tratado['Data'], format='%d%m%Y', errors='coerce')
data_ouro_tratado = data_ouro_tratado.dropna(how='any')
data_ouro_tratado = data_ouro_tratado.sort_values('Data')

#Dólar - organização e tratamento das Datas do arquivo
data_BRL['Date'] = pd.to_datetime(data_BRL['Date'])
data_BRL = data_BRL.sort_values('Date')
data_BRL.rename(columns={'Date': 'Data'}, inplace={True})

#Petróleo - organização e tratamento das Datas do arquivo
data_petroleo_tratado['Data'] = pd.to_datetime(data_petroleo_tratado['Data'],  format='%d%m%Y', errors='coerce')
data_petroleo_tratado = data_petroleo_tratado.sort_values('Data')
data_petroleo_tratado = data_petroleo_tratado.dropna(how='any')


### Checa por valores nulos/etc nas colunas ###
print(data_GOL.isnull().any())
print(data_bvsp.isnull().any())
print(data_ouro_tratado.isnull().any())
print(data_BRL.isnull().any())
print(data_petroleo_tratado.isnull().any())

###### Construindo X e y ######

X = data_GOL[['Data']] #Define a data como primeira coluna para pegar apenas os valores correspondentes a estas datas
X = X.merge(data_GOL[['Data', 'Close']], on='Data', how='inner')
X = X.merge(data_ouro_tratado[['Data', 'Ultimo']], on='Data', how='inner')
X = X.merge(data_petroleo_tratado[['Data', 'Ultimo']], on='Data', how='inner')
X = X.merge(data_bvsp[['Data', 'Close']], on='Data', how='inner')
X = X.merge(data_BRL[['Data', 'Close']], on='Data', how='inner')

X.columns = ['Data', 'Close_GOL', 'Ultimo_ouro', 'Ultimo_petroleo', 'Ultimo_bovespa', 'Ultimo_dólar']
X = X.dropna(how='any')#Drop em qualquer entrada que tenha valor nulo
Z = X
print(X)

#Definição de y, após definição de X para garantir que o tamanho de X e y sejam os mesmos, e obedeçam às datas em todas as entradas 
y = X['Close_GOL']

# Obtendo apenas as colunas das características
X = X[['Ultimo_ouro', 'Ultimo_petroleo', 'Ultimo_bovespa', 'Ultimo_dólar']]

# Definição da massa de teste e treino
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size= 0.25)

linear_regression = LinearRegression()

#Aplicando regressão linear nos dados de treino
linear_regression.fit(X_treino, y_treino)
y_prev = linear_regression.predict(X_teste)


#Respostas
print(y_prev)
print(np.sqrt(metrics.mean_squared_error(y_teste, y_prev)))


#Representação visual

sns.pairplot(Z, x_vars=['Ultimo_ouro', 'Ultimo_petroleo', 'Ultimo_bovespa', 'Ultimo_dólar'], y_vars='Close_GOL', height=5, kind='reg')

plt.show()
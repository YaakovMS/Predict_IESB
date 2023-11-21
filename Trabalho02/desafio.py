import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def tratar_arquivo_csv(arquivo):
    # Importando o arquivo CSV
    df = pd.read_csv(arquivo)

    # Convertendo a colunas de data para o formato datetime64
    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True)


    # Removendo as aspas duplas das colunas de texto
    df["Último"] = df["Último"].str.replace("\"", "")
    df["Abertura"] = df["Abertura"].str.replace("\"", "")
    df["Máxima"] = df["Máxima"].str.replace("\"", "")
    df["Mínima"] = df["Mínima"].str.replace("\"", "")
    df["Var%"] = df["Var%"].str.replace("\"", "")

    # Substituindo os pontos e vírgulas das colunas de números por pontos
    df["Último"] = df["Último"].str.replace(",", ".")
    df["Abertura"] = df["Abertura"].str.replace(",", ".")
    df["Máxima"] = df["Máxima"].str.replace(",", ".")
    df["Mínima"] = df["Mínima"].str.replace(",", ".")
    df["Vol."] = df["Vol."].str.replace(",", ".")
    df["Var%"] = df["Var%"].str.replace(",", ".")

    # Atribuindo os nomes das colunas conforme desejado
    df.columns = ["Data", "Ultimo", "Abertura", "Maxima", "Minima", "Vol.", "Var%"]

    return df


#Importando arquivos de dados
data_bvsp = pd.read_csv('./dados/^BVSP.csv')
data_BRL = pd.read_csv('./dados/BRL=X.csv')
data_GOL = pd.read_csv('./dados/GOLL4.SA.csv')

#Importando e tratando arquivos de dados não tratados
data_ouro_tratado = tratar_arquivo_csv('./dados/DESAFIO - Ouro Futuros Dados Históricos.csv')
data_petroleo_tratado = tratar_arquivo_csv('./dados/DESAFIO - Petróleo WTI Futuros Dados Históricos.csv')


#Tratamento fino dos dados de ouro
data_ouro_tratado["Ultimo"] = data_ouro_tratado["Ultimo"].str.replace(".", "", n=1)
data_ouro_tratado["Abertura"] = data_ouro_tratado["Abertura"].str.replace(".", "", n=1)
data_ouro_tratado["Maxima"] = data_ouro_tratado["Maxima"].str.replace(".", "", n=1)
data_ouro_tratado["Minima"] = data_ouro_tratado["Minima"].str.replace(".", "", n=1)
# Convertendo colunas para tipo float
data_ouro_tratado['Var%'] = data_ouro_tratado['Var%'].str.rstrip('%').astype('float') / 100.0
data_ouro_tratado[["Ultimo", 'Abertura', 'Maxima', 'Minima']] = data_ouro_tratado[["Ultimo", 'Abertura', 'Maxima', 'Minima']].astype(float)

#Tratamento fino dos dados de petroleo
data_petroleo_tratado["Ultimo"] = data_petroleo_tratado["Ultimo"].str.replace(",", ".", n=1)
data_petroleo_tratado["Abertura"] = data_petroleo_tratado["Abertura"].str.replace(",",".", n=1)
data_petroleo_tratado["Maxima"] = data_petroleo_tratado["Maxima"].str.replace(",", ".", n=1)
data_petroleo_tratado["Minima"] = data_petroleo_tratado["Minima"].str.replace(",", ".", n=1)
#Convertendo colunas para tipo float
data_petroleo_tratado['Var%'] = data_petroleo_tratado['Var%'].str.rstrip('%').astype(float) / 100.0
data_petroleo_tratado[["Ultimo", 'Abertura', 'Maxima', 'Minima']] = data_petroleo_tratado[["Ultimo", 'Abertura', 'Maxima', 'Minima']].astype(float)



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

data_ouro_tratado = data_ouro_tratado.dropna(how='any')
data_ouro_tratado = data_ouro_tratado.sort_values('Data')


#Dólar - organização e tratamento das Datas do arquivo
data_BRL['Date'] = pd.to_datetime(data_BRL['Date'])
data_BRL = data_BRL.sort_values('Date')
data_BRL.rename(columns={'Date': 'Data'}, inplace={True})

#Petróleo - organização e tratamento das Datas do arquivo
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

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Função para atualiza o DataFrame com novos dados
def update_dataframe(df, new_data):
    # Converte a coluna 'Data' para datetime
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
    new_data['Data'] = pd.to_datetime(new_data['Data'], dayfirst=True)

    # Encontra a data mais recente no DataFrame existente
    last_date = df['Data'].max()

    # Filtra as novas linhas que são mais recentes do que a última data
    new_rows = new_data[new_data['Data'] > last_date]

    # Concatena os novos dados com o DataFrame existente se houver novas linhas
    if not new_rows.empty:
        updated_df = pd.concat([df, new_rows], ignore_index=True)
    else:
        updated_df = df
    return updated_df

# URL do site IPEADATA
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'

# Faz uma requisição GET ao site e captura a resposta
response = requests.get(url)

# Verifica se a requisição foi bem sucedida
if response.status_code == 200:
    # Cria um objeto BeautifulSoup para analisar o HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    # Procura pela tabela no HTML analisado
    table = soup.find('table', {'id': 'grd_DXMainTable'})
    # Usa o pandas para ler a tabela HTML diretamente para um DataFrame
    new_df = pd.read_html(str(table), header=0)[0]

    # Verifica se o arquivo do DataFrame existe e carrega, ou cria um novo DataFrame se não existir
    path = '/content/ipea.csv'
    try:
        existing_df = pd.read_csv(path)
    except FileNotFoundError:
        existing_df = new_df # Se o arquivo não existir, considere os dados atuais como o DataFrame existente

    # Atualiza o DataFrame existente com novos dados (Carga Incremental)
    updated_df = update_dataframe(existing_df, new_df)

    updated_df['Preço - petróleo bruto - Brent (FOB)'] = updated_df['Preço - petróleo bruto - Brent (FOB)']/100

    # Salva o DataFrame atualizado para o arquivo
    updated_df.to_csv(path, index=False)

    # Mostra as primeiras linhas do DataFrame atualizado
    updated_df.head()
else:
    print('Falha ao acessar a página: Status code', response.status_code)

# Importação Biblioteca,

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt

# Função para ler e preparar os dados.

def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by='Data', ascending=True)
    df.set_index('Data', inplace=True)
    return df

# Função para testar a estacionariedade.

def test_stationarity(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

# Função para encontrar os melhores parâmetros ARIMA.

def optimize_arima(series):
    auto_model = auto_arima(series, start_p=0, start_q=0,
                            max_p=10, max_q=10, m=12,
                            seasonal=False,
                            d=None, trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
    return auto_model

# Função para calcular RMSE.

def calculate_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# Carregar e preparar os dados.

file_path = '/content/ipea.csv'
df = prepare_data(file_path)
price_series = df['Preço - petróleo bruto - Brent (FOB)']

# Testar estacionariedade

test_stationarity(price_series)

# Encontrar os melhores parâmetros para o ARIMA.

optimized_arima = optimize_arima(price_series)

# Dividir os dados para treino e teste.

train_size = len(price_series) - 15
train, test = price_series[0:train_size], price_series[train_size:]

# Ajustar o modelo nos dados de treino.

model_fit = optimized_arima.fit(train)

# Fazer previsões.

predictions = model_fit.predict(n_periods=len(test))

# Calcular RMSE.

rmse = calculate_rmse(test, predictions)

# Exibir os resultados.

print(f"Melhores parâmetros ARIMA: {model_fit.order}")
print(f"RMSE: {rmse}")

# Plotar os valores reais vs. previsões.

plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Treino')
plt.plot(test.index, test, label='Teste Real')
plt.plot(test.index, predictions, label='Previsões', color='red')
plt.title('Preço do Petróleo Bruto - Brent: Real vs Previsões')
plt.legend()
plt.show()


import streamlit as st
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])

# Função para encontrar os melhores parâmetros ARIMA.
def optimize_arima(series):
    auto_model = auto_arima(series, start_p=0, start_q=0,
                            max_p=5, max_q=5, m=6,
                            seasonal=False,
                            d=None, trace=False,  # trace=True para ver o progresso
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)
    return auto_model

# Função para calcular RMSE.
def calculate_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# Carregar e preparar os dados.
file_path = 'ipea.csv'  # Substitua pelo caminho correto se necessário
df = prepare_data(file_path)
price_series = df['Preço - petróleo bruto - Brent (FOB)']

# Interface do Streamlit
st.title('Previsão do Preço do Petróleo Bruto - Brent')
train_size = st.slider('Escolha a quantidade de dias para treinamento:', 
                        min_value=30, max_value=len(price_series)-15, value=365, step=30)
n_periods = st.number_input('Escolha a quantidade de dias para previsão:', 
                             min_value=1, max_value=60, value=15)

if st.button('Treinar Modelo'):
    # Dividir os dados para treino e teste.
    train, test = price_series[:-n_periods], price_series[-n_periods:]

    # Ajustar o modelo nos dados de treino.
    st.write('Treinando o modelo...')
    optimized_arima_model = optimize_arima(train)
    model_fit = optimized_arima_model.fit(train)

    # Fazer previsões.
    predictions = model_fit.predict(n_periods=n_periods)

    # Calcular RMSE.
    rmse = calculate_rmse(test, predictions)
    st.write(f"Melhores parâmetros ARIMA: {model_fit.order}")
    st.write(f"RMSE: {rmse}")

    # Plotar os valores reais vs. previsões.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train.index, train, label='Treino')
    ax.plot(test.index, test, label='Teste Real')
    ax.plot(test.index, predictions, label='Previsões', color='red')
    ax.set_title('Preço do Petróleo Bruto - Brent: Real vs Previsões')
    ax.legend()
    st.pyplot(fig)

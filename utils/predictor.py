import numpy as np
import pandas as pd
import cloudpickle
from sklearn.preprocessing import StandardScaler

def carregar_modelo_e_scaler():
    with open("modelo_lstm.pkl", "rb") as f:
        modelo = cloudpickle.load(f)
    with open("scaler_lstm.pkl", "rb") as f:
        scaler = cloudpickle.load(f)
    return modelo, scaler

def preparar_features(df):
    df["mm5"] = df["Fechamento"].rolling(window=5).mean()
    df["mm10"] = df["Fechamento"].rolling(window=10).mean()
    df["retorno_dia"] = df["Fechamento"].pct_change(fill_method=None) * 100
    df["volatilidade"] = df["Fechamento"].rolling(window=5).std()

    df["mm5"] = df["mm5"].fillna(df["Fechamento"])
    df["mm10"] = df["mm10"].fillna(df["Fechamento"])
    df["retorno_dia"] = df["retorno_dia"].fillna(0)
    df["volatilidade"] = df["volatilidade"].fillna(0)

    return df.reset_index(drop=True)

def gerar_previsao(df_base, modelo, scaler, dias_prever=7):
    features = ["Abertura", "Maxima", "Minima", "Volume", "Variacao", "mm5", "mm10", "retorno_dia", "volatilidade"]

    X_scaled = scaler.transform(df_base[features + ["Fechamento"]])
    janela = 10
    entrada = X_scaled[-janela:, :-1]

    previsoes = []
    datas = []
    ultima_data = df_base["Data"].iloc[-1]

    for i in range(int(dias_prever)):
        entrada_3d = np.expand_dims(entrada, axis=0)
        pred_norm = modelo.predict(entrada_3d, verbose=0)

        linha_falsa = np.zeros(X_scaled.shape[1])
        linha_falsa[:-1] = entrada[-1]
        linha_falsa[-1] = pred_norm[0]

        linha_inversa = scaler.inverse_transform([linha_falsa])
        previsao_real = linha_inversa[0][-1]

        previsoes.append(previsao_real)
        datas.append(ultima_data + pd.Timedelta(days=i+1))

        nova_entrada = np.append(entrada[1:], [linha_falsa[:-1]], axis=0)
        entrada = nova_entrada

    df_previsao = pd.DataFrame({"Data": datas, "Preco_Previsto": previsoes})
    return df_previsao

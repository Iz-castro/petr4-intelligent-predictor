import pandas as pd
import os

def carregar_historico_insercoes():
    if os.path.exists("logs/dados_pos_treino.csv"):
        df = pd.read_csv("logs/dados_pos_treino.csv")
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        return df.sort_values("Data")
    else:
        return pd.DataFrame()

def carregar_historico_previsoes():
    if os.path.exists("logs/previsao_futura.csv"):
        df = pd.read_csv("logs/previsao_futura.csv")
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        return df.sort_values("Data")
    else:
        return pd.DataFrame()

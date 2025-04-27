import pandas as pd
import os

# Função para corrigir o volume (K / M)
def corrigir_volume(val):
    val = str(val).replace(".", "").replace(",", ".")
    if "K" in val:
        return float(val.replace("K", "")) * 1e3
    elif "M" in val:
        return float(val.replace("M", "")) * 1e6
    else:
        return float(val)

# Função de pré-processamento principal
def preprocess_dataframe(df):
    df["Data"] = pd.to_datetime(df["Data"])
    for col in ["Fechamento", "Abertura", "Maxima", "Minima"]:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    df["Volume"] = df["Volume"].apply(corrigir_volume)
    df["Variacao"] = df["Variacao"].astype(str).str.replace(",", ".").str.replace("%", "").astype(float)

    return df.sort_values("Data").reset_index(drop=True)

# Função para adicionar as features de LSTM
def add_features_lstm(df):
    df["mm5"] = df["Fechamento"].rolling(window=5).mean()
    df["mm10"] = df["Fechamento"].rolling(window=10).mean()
    df["retorno_dia"] = df["Fechamento"].pct_change() * 100
    df["volatilidade"] = df["Fechamento"].rolling(window=5).std()

    return df.dropna().reset_index(drop=True)

# Função para carregar a base histórica
def carregar_base_original():
    df = pd.read_csv("logs/historico_pet4_2000_2025.csv")
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    return df.sort_values("Data").dropna()

# Função para carregar dados pós-treino
def carregar_dados_inseridos():
    if os.path.exists("logs/dados_pos_treino.csv"):
        df = pd.read_csv("logs/dados_pos_treino.csv")
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        return df.sort_values("Data").dropna()
    else:
        return pd.DataFrame()

# Função para salvar novos preços diários
def salvar_novo_preco(df_novo):
    os.makedirs("logs", exist_ok=True)
    path = "logs/dados_pos_treino.csv"
    if os.path.exists(path):
        df_existente = pd.read_csv(path)
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo
    df_final.to_csv(path, index=False)

# Função para salvar previsões futuras
def salvar_previsao_futura(df_previsao):
    os.makedirs("logs", exist_ok=True)
    df_previsao.to_csv("logs/previsao_futura.csv", index=False)

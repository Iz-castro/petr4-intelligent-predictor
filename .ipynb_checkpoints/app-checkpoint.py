# app.py
import streamlit as st
from utils.data_handler import (
    carregar_base_original,
    carregar_dados_inseridos,
    salvar_novo_preco,
    salvar_previsao_futura
)
from utils.predictor import (
    carregar_modelo_e_scaler,
    preparar_features,
    gerar_previsao
)
from utils.historical_manager import (
    carregar_historico_insercoes,
    carregar_historico_previsoes
)
from utils.openai_logic import gerar_recomendacao_chat

from utils.data_handler import corrigir_volume

import numpy as np
import pandas as pd
import os
import json
from datetime import date, datetime

# Configuração da página
st.set_page_config(page_title="PETR4 Inteligente", layout="wide")
st.title("📈 Análise Inteligente da PETR4 com IA")

# === Carregar usuário ===
usuario = {}
usuario_path = "usuario.json"
if os.path.exists(usuario_path):
    with open(usuario_path, "r") as f:
        usuario = json.load(f)

# TABS
tabs = st.tabs([
    "👤 Perfil do Investidor",
    "🔎 Recomendação IA",
    "📥 Inserir Preço Diário",
    "📈 Histórico de Preços",
    "🔮 Previsão Futura",
    "📜 Histórico de Previsões"
])

# === TABS[0] - Perfil do Investidor ===
with tabs[0]:
    st.subheader("👤 Cadastro do Investidor")

    nome = st.text_input("Nome do usuário", value=usuario.get("nome", ""))
    perfil = st.selectbox(
        "Perfil do investidor", 
        ["buy_and_hold", "swing_trade", "day_trade"],
        index=["buy_and_hold", "swing_trade", "day_trade"].index(usuario.get("perfil", "buy_and_hold"))
    )
    preco_medio = st.number_input("Preço médio atual por ação (R$)", min_value=0.0, value=usuario.get("preco_medio", 30.0))
    qtde_atual = st.number_input("Quantidade atual de ações", min_value=0, value=usuario.get("qtde_atual", 100))
    qtde_desejada = st.number_input("Meta de ações desejadas", min_value=0, value=usuario.get("qtde_desejada", 200))

    if st.button("Salvar Perfil"):
        perfil_data = {
            "nome": nome,
            "perfil": perfil,
            "preco_medio": preco_medio,
            "qtde_atual": qtde_atual,
            "qtde_desejada": qtde_desejada
        }
        with open(usuario_path, "w") as f:
            json.dump(perfil_data, f)
        st.success("✅ Perfil salvo com sucesso!")

# === TABS[1] - Recomendação IA ===
with tabs[1]:
    st.subheader("🔎 Recomendação Inteligente com IA")

    try:
        df_base = carregar_base_original()
        df_inserido = carregar_dados_inseridos()
        if not df_inserido.empty:
            df_base = pd.concat([df_base, df_inserido], ignore_index=True)
            df_base = df_base.sort_values("Data").dropna()
    except Exception as e:
        st.error(f"Erro ao carregar base de dados: {e}")
        st.stop()

    preco_atual = df_base["Fechamento"].iloc[-1]

    preco_temporario = st.number_input("💡 Preço temporário para simulação (não será salvo)", min_value=0.0, step=0.01)
    if preco_temporario > 0:
        preco_atual = preco_temporario

    objetivo = st.text_input("🎯 Objetivo do investidor", value="acumular dividendos")

    if st.button("🤖 Gerar Recomendação com OpenAI"):
        resposta = gerar_recomendacao_chat(
            df_base,
            usuario["perfil"],
            usuario["preco_medio"],
            usuario["qtde_atual"],
            usuario["qtde_desejada"],
            objetivo,
            preco_atual,
            usuario_nome=usuario.get("nome", "Usuário")
        )
        st.markdown(f"### 🧠 Recomendação para {usuario.get('nome', 'Usuário')}")
        st.info(resposta)


# === TABS[2] - Inserir Preço Diário  ===
with tabs[2]:
    st.subheader("📥 Inserir novo preço diário (mercado)")

    data_limite = datetime(2025, 4, 24)
    nova_data = st.date_input("Data", min_value=data_limite, value=date.today())
    novo_fechamento = st.number_input("💵 Fechamento (R$)", min_value=0.0, step=0.01)

    if st.button("Salvar Preço Diário"):
        try:
            df_base = carregar_base_original()
            df_base = df_base.sort_values("Data").dropna()

            # Inicialmente assume o último registro da base histórica
            ultimo_registro = df_base.iloc[-1]

            # Se existir dados pós treino, usar o mais recente
            if os.path.exists("logs/dados_pos_treino.csv"):
                df_inserido = pd.read_csv("logs/dados_pos_treino.csv")
                if not df_inserido.empty and "Fechamento" in df_inserido.columns:
                    df_inserido["Data"] = pd.to_datetime(df_inserido["Data"], errors="coerce")
                    df_inserido = df_inserido.sort_values("Data").dropna()
                    ultimo_registro = df_inserido.iloc[-1]
                else:
                    st.warning("⚠️ Arquivo de inserções vazio ou sem coluna 'Fechamento'. Usando base histórica.")
            else:
                st.warning("⚠️ Nenhum dado pós-treino encontrado. Usando base histórica.")

            # Garantir que os tipos sejam float
            ultimo_fechamento = float(ultimo_registro["Fechamento"])
            ultimo_volume = float(ultimo_registro["Volume"])

            novo_registro = {
                "Data": nova_data,
                "Abertura": ultimo_fechamento,
                "Maxima": novo_fechamento * 1.01,
                "Minima": novo_fechamento * 0.99,
                "Fechamento": novo_fechamento,
                "Volume": ultimo_volume,
                "Variacao": (novo_fechamento - ultimo_fechamento) / ultimo_fechamento * 100
            }

            df_novo = pd.DataFrame([novo_registro])
            salvar_novo_preco(df_novo)
            st.success("✅ Preço salvo com sucesso!")

        except Exception as e:
            st.error(f"Erro ao salvar preço diário: {e}")


# === TABS[3] - Histórico de Preços Inseridos ===
with tabs[3]:
    st.subheader("📈 Histórico de preços inseridos")

    df_historico = carregar_historico_insercoes()

    if not df_historico.empty:
        st.dataframe(df_historico.sort_values("Data", ascending=False))

        if "Fechamento" in df_historico.columns:
            try:
                st.line_chart(df_historico.set_index("Data")["Fechamento"])
            except Exception as e:
                st.warning(f"⚠️ Não foi possível gerar o gráfico: {e}")
        else:
            st.warning("⚠️ Dados inseridos ainda não possuem coluna 'Fechamento' para exibir gráfico.")
    else:
        st.info("Nenhum dado registrado ainda.")


# === TABS[4] - Previsão Futura Profissional ===
with tabs[4]:
    st.subheader("🔮 Previsão Futura com LSTM")

    dias_prever = st.number_input("Quantos dias deseja prever?", min_value=1, max_value=60, value=7)

    if st.button("Gerar Previsão Futura"):
        try:
            from utils.data_handler import preprocess_dataframe, add_features_lstm
            from utils.predictor import carregar_modelo_e_scaler
            import matplotlib.pyplot as plt

            def proximo_dia_util(data_atual):
                data_atual += pd.Timedelta(days=1)
                while data_atual.weekday() >= 5:  # 5 = sábado (5) ou domingo (6)
                    data_atual += pd.Timedelta(days=1)
                return data_atual

            # 1. Carregar histórico base
            df_base = carregar_base_original()
            df_inserido = carregar_dados_inseridos()
            if not df_inserido.empty:
                df_base = pd.concat([df_base, df_inserido], ignore_index=True)

            df_base = df_base.sort_values("Data").dropna()

            # 2. Pré-processar dados como no notebook de treino
            df_base = preprocess_dataframe(df_base)
            df_base = add_features_lstm(df_base)

            # 3. Carregar modelo e scaler
            modelo, scaler = carregar_modelo_e_scaler()

            # 4. Normalizar dados
            features = ["Abertura", "Maxima", "Minima", "Volume", "Variacao", "mm5", "mm10", "retorno_dia", "volatilidade"]
            window_size = 10
            scaled = scaler.transform(df_base[features + ["Fechamento"]])

            # Pegar última janela para previsão
            entrada = scaled[-window_size:, :-1]

            previsoes = []
            datas = []
            ultima_data = df_base["Data"].iloc[-1]

            for _ in range(int(dias_prever)):
                entrada_3d = np.expand_dims(entrada, axis=0)
                pred_norm = modelo.predict(entrada_3d, verbose=0)
            
                dummy_features = entrada[-1]    # 9 features
                dummy_target = pred_norm[0][0]  # Corrigido: pegar o número
            
                linha_completa = np.concatenate([dummy_features, [dummy_target]])  # Agora sim concatena 9+1
                linha_inversa = scaler.inverse_transform([linha_completa])
            
                previsao_real = linha_inversa[0, -1]
            
                previsoes.append(previsao_real)
                ultima_data = proximo_dia_util(ultima_data)
                datas.append(ultima_data)
            
                nova_entrada = np.append(entrada[1:], [linha_completa[:-1]], axis=0)
                entrada = nova_entrada

            # 5. Salvar previsões
            df_previsao = pd.DataFrame({"Data": datas, "Preco_Previsto": previsoes})
            salvar_previsao_futura(df_previsao)

            # 6. Plotar gráfico profissional
            ultimos_reais = df_base[["Data", "Fechamento"]].set_index("Data").tail(30)
            ultimos_reais = ultimos_reais.rename(columns={"Fechamento": "Preço Real"})

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(ultimos_reais.index, ultimos_reais["Preço Real"], label="Preço Real", color="blue", linewidth=2)
            ax.plot(df_previsao.set_index("Data").index, df_previsao["Preco_Previsto"], label="Preço Previsto", linestyle="--", color="orange", linewidth=2)

            ax.set_title("📈 Previsão do Preço de Fechamento PETR4", fontsize=16)
            ax.set_xlabel("Data", fontsize=12)
            ax.set_ylabel("Preço (R$)", fontsize=12)
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

            # Mostrar também o DataFrame de previsão
            st.dataframe(df_previsao.reset_index(drop=True))

        except Exception as e:
            st.error(f"Erro ao gerar previsão: {e}")

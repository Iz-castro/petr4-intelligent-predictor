import openai
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gerar_recomendacao_chat(df, perfil, preco_medio, qtde_atual, qtde_desejada,
                             objetivo="acumular dividendos", preco_atual=None, usuario_nome="Usuário"):
    resumo_df = df.tail(3).to_string(index=False)

    prompt = f"""
Você é um assistente financeiro especializado em análise de ações, focando na Petrobras (PETR4).

Informações do investidor:
- Nome: **{usuario_nome}**
- Perfil: **{perfil}**
- Quantidade atual: **{qtde_atual} ações**
- Preço médio: **R${preco_medio:.2f}**
- Meta de ações: **{qtde_desejada} ações**
- Objetivo: **{objetivo}**

Preço atual informado da ação PETR4: **R${preco_atual:.2f}**

Resumo das últimas previsões geradas pelo modelo LSTM:
{resumo_df}

Com base nessas informações, forneça:
1. Uma análise da tendência atual;
2. Sugestões de compra, manutenção ou venda;
3. Estratégias adequadas ao perfil do investidor.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{usuario_nome.replace(' ', '_')}_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {usuario_nome}\n")
        f.write("🔹 PROMPT:\n" + prompt + "\n")
        f.write("🔸 RESPOSTA:\n" + content + "\n")
        f.write("-" * 60 + "\n")

    return content

# 📈 PETR4 Intelligent Predictor - Versão 0.1.0

Projeto de previsão inteligente para o preço de fechamento da ação PETR4 utilizando Redes Neurais LSTM.

---

## 🚀 Funcionalidades

- 🔹 Pré-processamento realista de dados históricos (corrige formatos de preços, volume, variação percentual);
- 🔹 Engenharia de features avançada (mm5, mm10, volatilidade, retorno diário);
- 🔹 Normalização automática dos dados para uso em modelos LSTM;
- 🔹 Previsão de preços futuros respeitando o calendário da B3 (não prevê sábados e domingos);
- 🔹 Interface amigável via Streamlit;
- 🔹 Gráfico Real vs Previsto com Matplotlib;
- 🔹 Armazenamento do histórico pós-treino para melhoria contínua.

---

## 📚 Tecnologias Utilizadas

- Python 3.10
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Cloudpickle

---

## 🛠️ Como Rodar o Projeto

1. Clone este repositório:

```
git clone https://github.com/seu-usuario/petr4-intelligent-predictor.git
```

2 . Crie e ative um ambiente virtual:

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```
pip install -r requirements.txt
```

4. Rode a aplicação:

```
streamlit run app.py
```

---

## 📂 Estrutura do Projeto

```
|--app.py                     # Aplicativo principal Streamlit
|--utils/
|------data_handler.py        # Funções de manipulação de dados
|------predictor.py           # Funções de modelagem e previsão
|------historical_manager.py  # Gerenciamento de históricos
|------openai_logic.py        # Integração com API OpenAI (opcional)
|--logs/                      # Histórico de inserções e previsões
|--modelo_lstm.pkl            # Modelo LSTM treinado
|--scaler_lstm.pkl            # Scaler usado na normalização
|--resultados_petr4_lstm.csv  # Resultados de treino
|--requirements.txt           # Dependências do projeto
|--README.md                  # Documentação do projeto
```

---

## 📅 Próximas Atualizações Planejadas

- 🔹 📥 Upload de CSV de novos dados para melhorar previsões futuras;

- 🔹 🗓️ Ajuste de feriados no calendário da B3;

- 🔹 📊 Melhorias na visualização dos dados e métricas adicionais;

- 🔹 🧠 Aperfeiçoamento do modelo com novos dados (retrain);

- 🔹 📦 Criação de pacotes para distribuição.

---

## 📢 Status

Versão Atual: 0.1.0
Status: Em desenvolvimento contínuo 🚧


---

## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.

---

## ✉️ Contato

Para dúvidas, sugestões ou colaborações:
**Izael Castro**  
Email: *izaeldecastro@gmail.com*
Github: *Iz-castro*

---


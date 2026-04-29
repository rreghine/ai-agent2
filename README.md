# E-Commerce AI Agent — Claude Edition | Parte 2

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![Claude](https://img.shields.io/badge/Anthropic-Claude_Sonnet-black?style=flat-square)
![Gemma](https://img.shields.io/badge/Google-Gemma_3_%7C_4-orange?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-Text--to--SQL-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Multi--LLM-red?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

Agente de IA para análise de e-commerce com suporte a 3 modelos de linguagem —
Anthropic Claude Sonnet, Google Gemma 3 27B e Google Gemma 4 31B. Utiliza
Text-to-SQL sobre banco SQLite real, avaliação híbrida com Ground Truth dinâmico
e LLM como Juiz, retry automático de SQL, rastreamento com MLflow e benchmark
ao vivo onde os 3 modelos respondem as mesmas perguntas sobre o mesmo banco
de dados para comparação justa de acurácia, tokens, latência e custo.

---

## Preview

### Interface do Agente
![Agente](preview_agente.png)

### Benchmark Multi-LLM
![Benchmark](preview_benchmark.png)

### Dashboard com Insights dos 3 Modelos
![Dashboard](preview_dashboard.png)

---

## Evolucoes em Relacao a Parte 1

| Componente | Parte 1 | Parte 2 |
|---|---|---|
| LLM principal | Google Gemma 3 | Anthropic Claude Sonnet |
| Dados | CSV + RAG + FAISS | SQLite + Text-to-SQL |
| Avaliacao | Overlap de palavras | Numerico + LLM como Juiz |
| Custo | Estimado | Real — usage API |
| Retry | Nao | Sim — SQL corrigido automaticamente |
| Observabilidade | MLflow | MLflow |
| Modelos | 1 | 3 — Gemma 3, Gemma 4, Claude |
| Abas | 2 | 4 — Agente, Metricas, Benchmark, Dashboard |

---

## Arquitetura

```
Pergunta do usuario
        |
Guardrails — valida escopo, dados sensiveis e tamanho
        |
Text-to-SQL — Claude converte pergunta em SQL
        |
SQLite — executa query nos dados reais (zero alucinacao em numeros)
        |
Claude — interpreta resultado em portugues
        |
Avaliador Hibrido
    |-- Numerico: erro relativo <= 2% correta / <= 10% parcial
    |-- LLM Juiz: Claude avalia semanticamente a propria resposta
        |
MLflow — loga tokens reais, custo, latencia, score de alucinacao
        |
Streamlit — 4 abas
    |-- Agente: chat com avaliacao inline
    |-- Metricas: dashboard da sessao
    |-- Benchmark: 3 modelos, mesma pergunta, comparacao ao vivo
    |-- Dashboard: insights gerados pelos 3 modelos simultaneamente
```

---

## Dataset

**Brazilian E-Commerce — Olist (Kaggle)**

- 99.441 pedidos reais e anonimizados
- 7 tabelas relacionadas — pedidos, clientes, produtos, vendedores, pagamentos, avaliacoes
- Periodo: 2016 a 2018
- Contexto 100% brasileiro

[Download do dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Estrutura do Repositorio

```
ai-agent-part2/
|
|-- Agent_Claude_SQL.ipynb        <- Notebook principal (Google Colab)
|-- app.py                        <- App Streamlit Multi-LLM
|-- olist.db                      <- Banco SQLite (gerado no notebook)
|-- requirements.txt
|-- preview_agente.png
|-- preview_benchmark.png
|-- preview_dashboard.png
|-- README.md
```

---

## Componentes

### Guardrails
Validacao em dois niveis — entrada e saida:

- Validacao de escopo — bloqueia perguntas fora do dominio de e-commerce
- Protecao de dados — impede exposicao de dados pessoais (CPF, senhas)
- Controle de tamanho — minimo 5 e maximo 500 caracteres

### Text-to-SQL com Retry Automatico
Claude converte linguagem natural em SQL valido e executa no banco:

```
Pergunta -> Claude gera SQL -> SQLite executa
                |
           Erro de SQL?
                |
         Claude corrige -> SQLite reexecuta (max 2 tentativas)
```

Vantagem sobre RAG: zero alucinacao em numeros — o valor vem diretamente do banco.

### Avaliador Hibrido

**Avaliacao Numerica** — para perguntas com resposta quantitativa:

| Erro Relativo | Status |
|---|---|
| <= 2% | Correta |
| <= 10% | Parcial |
| > 10% | Alucinacao |

**LLM como Juiz** — para perguntas abertas:

Claude recebe a pergunta, o ground truth e a resposta do agente e avalia semanticamente retornando score de 0 a 1 com justificativa.

### Ground Truth Dinamico
Calculado diretamente no banco via SQL — nao e dado fixo:

- Ticket medio: `SELECT ROUND(AVG(price),2) FROM items`
- Taxa de atraso: calculada com comparacao de datas reais
- Estado com mais pedidos: contagem real por estado
- 8 tipos de perguntas com ground truth automatico

### Benchmark ao Vivo
Os 3 modelos recebem as mesmas perguntas, usam o mesmo banco SQLite e o mesmo metodo Text-to-SQL. A unica variavel e o modelo de linguagem.

**Modelos disponíveis:**

| Modelo | API | Custo |
|---|---|---|
| Google Gemma 3 27B | Google AI Studio | Gratuito |
| Google Gemma 4 31B | Google AI Studio | Gratuito |
| Anthropic Claude Sonnet | Anthropic API | $3/1M tokens input |

### Dashboard com Insights dos 3 Modelos
Os 3 modelos analisam os mesmos dados e geram 5 insights estrategicos independentes. O usuario ve a diferenca de qualidade, profundidade e custo em tempo real.

---

## Como Executar

```bash
# 1. Clonar o repositorio
git clone https://github.com/rreghine/ai-agent-part2.git
cd ai-agent-part2

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar API Keys
# Crie o arquivo .streamlit/secrets.toml
echo 'ANTHROPIC_API_KEY = "sua_key_aqui"' >> .streamlit/secrets.toml
echo 'GEMINI_API_KEY    = "sua_key_aqui"' >> .streamlit/secrets.toml

# 4. Rodar o app
streamlit run app.py
```

### Obtendo as API Keys

**Anthropic (Claude):**
1. Acesse [console.anthropic.com](https://console.anthropic.com)
2. Va em API Keys -> Create Key
3. Adicione creditos — minimo $5 suficiente para o projeto inteiro

**Google (Gemma 3 e Gemma 4):**
1. Acesse [aistudio.google.com](https://aistudio.google.com)
2. Clique em Get API Key -> Create API Key
3. Gratuito — sem necessidade de cartao de credito

### Rodando no Google Colab

```python
import shutil, os, subprocess, threading, time
from google.colab import userdata

shutil.copy('/content/drive/MyDrive/.../app.py', '/content/app.py')
shutil.copy('/content/drive/MyDrive/.../olist.db', '/content/olist.db')

os.makedirs('/content/.streamlit', exist_ok=True)
with open('/content/.streamlit/secrets.toml', 'w') as f:
    f.write(f'ANTHROPIC_API_KEY = "{userdata.get("ANTHROPIC_API_KEY")}"\n')
    f.write(f'GEMINI_API_KEY    = "{userdata.get("GEMINI_API_KEY")}"\n')

def run():
    subprocess.run(['streamlit', 'run', '/content/app.py',
                    '--server.port', '8501', '--server.headless', 'true',
                    '--server.enableCORS', 'false'])

threading.Thread(target=run, daemon=True).start()
time.sleep(8)

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8501)"))
```

---

## Tecnologias Utilizadas

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3.12 |
| LLMs | Anthropic Claude Sonnet · Google Gemma 3 27B · Google Gemma 4 31B |
| Banco de dados | SQLite |
| Metodo | Text-to-SQL |
| Avaliacao | Ground Truth Hibrido + LLM como Juiz |
| Rastreamento | MLflow |
| Interface | Streamlit |
| Ambiente | Google Colab |

---

## Proximos Passos — Parte 3

- Langfuse — observabilidade profissional de LLMs
- RAGAS — framework de avaliacao especializado
- RAG leve com few-shot dinamico para melhorar o SQL
- Adicao de Mistral e Qwen ao benchmark

---

## Parte 1

O projeto original com Google Gemma 3, RAG + FAISS e avaliacao por overlap de palavras esta disponivel em:

[github.com/rreghine/ai-agent](https://github.com/rreghine/ai-agent)

---

## Autor

**Rafael Reghine Munhoz**
Data Analyst | Data Science & Analytics | MBA USP

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rafaelreghine-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/rafaelreghine)
[![GitHub](https://img.shields.io/badge/GitHub-rreghine-black?style=flat-square&logo=github)](https://github.com/rreghine)

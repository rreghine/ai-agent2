import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import sqlite3, time, json, os, re
from datetime import datetime

st.set_page_config(page_title="E-Commerce AI Agent | Multi-LLM",
                   page_icon="📊", layout="wide", initial_sidebar_state="expanded")

A1='#0a1628';A2='#1e3a5f';A3='#2e6fad';A4='#4a90c4';A5='#6baed6'
A6='#c6dbef';BR='#e8f1fa';WH='#ffffff';BD='#dce8f5';TX='#2c3e50'
TL='#7f8c8d';VR='#c0392b';AM='#e67e22'

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{{font-family:'IBM Plex Sans',sans-serif;background:{WH};color:{TX}}}
.main{{background:{WH}}}.block-container{{padding-top:1.5rem;padding-bottom:2rem}}
.header-wrap{{background:linear-gradient(135deg,{A1} 0%,{A2} 60%,{A3} 100%);padding:16px 22px 14px;border-radius:6px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between}}
.header-title{{font-size:1.15rem;font-weight:700;color:{WH}}}
.header-badge{{font-size:.62rem;font-weight:700;background:rgba(255,255,255,.2);color:{WH};border:1px solid rgba(255,255,255,.35);border-radius:3px;padding:2px 7px;letter-spacing:1px;margin-left:8px;vertical-align:middle}}
.header-sub{{font-size:.7rem;color:rgba(255,255,255,.55);margin-top:4px}}
.header-right{{font-size:.7rem;color:rgba(255,255,255,.45);text-align:right}}
.kpi-card{{background:{WH};border:1px solid {BD};border-top:3px solid var(--c);border-radius:6px;padding:14px 16px 12px;box-shadow:0 1px 4px rgba(10,22,40,.05)}}
.kpi-label{{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:1.4px;color:{TL};margin-bottom:6px}}
.kpi-value{{font-size:1.65rem;font-weight:700;line-height:1;color:var(--c);letter-spacing:-.5px}}
.kpi-sub{{font-size:.63rem;color:{TL};margin-top:4px}}
.sec-label{{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:{A4};border-bottom:2px solid {A6};padding-bottom:6px;margin:20px 0 14px}}
.resp-box{{background:{WH};border:1px solid {BD};border-left:4px solid {A3};border-radius:0 6px 6px 0;padding:16px 20px;font-size:.88rem;color:{TX};line-height:1.75}}
.guard-box{{background:#fff8f0;border:1px solid #fde8cc;border-left:4px solid {AM};border-radius:0 6px 6px 0;padding:14px 18px;font-size:.85rem;color:#7d4e1a}}
.eval-row{{display:flex;border:1px solid {BD};border-radius:6px;overflow:hidden;margin-top:10px}}
.eval-cell{{flex:1;padding:9px 13px;border-right:1px solid {BD};background:{BR}}}
.eval-cell:last-child{{border-right:none}}
.eval-lbl{{font-size:.57rem;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:{TL};margin-bottom:3px}}
.eval-val{{font-size:.82rem;font-weight:600;color:{TX}}}
.bench-card{{background:{WH};border:1px solid {BD};border-radius:8px;padding:20px;box-shadow:0 1px 6px rgba(10,22,40,.06)}}
.bench-title{{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:{TL};margin-bottom:12px}}
.insight-card{{background:{WH};border:1px solid {BD};border-left:4px solid {A3};border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:10px;font-size:.88rem;color:{TX};line-height:1.7;box-shadow:0 1px 4px rgba(10,22,40,.04)}}
.insight-num{{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{A4};margin-bottom:4px}}
.stTabs [data-baseweb="tab-list"]{{gap:0;border-bottom:2px solid {BD}}}
.stTabs [data-baseweb="tab"]{{font-size:.82rem;font-weight:600;padding:10px 22px;color:{TL}}}
.stTabs [aria-selected="true"]{{color:{A3}!important;border-bottom:2px solid {A3}!important}}
#MainMenu{{visibility:hidden}}footer{{visibility:hidden}}header{{visibility:hidden}}
</style>""", unsafe_allow_html=True)

# ── API Keys ───────────────────────────────────────────────────────────────
def get_secret(name):
    for fn in [lambda: os.environ.get(name),
               lambda: st.secrets.get(name),
               lambda: __import__('google.colab',fromlist=['userdata']).userdata.get(name)]:
        try:
            k = fn()
            if k: return k
        except: pass
    return None

# Anthropic (Claude)
try:
    import anthropic as _a
    _ak = get_secret('ANTHROPIC_API_KEY')
    claude_client = _a.Anthropic(api_key=_ak) if _ak else None
except: claude_client = None

# Google (Gemma 4)
try:
    from google import genai as _genai
    _gk = get_secret('GEMINI_API_KEY')
    gemma_client = _genai.Client(api_key=_gk) if _gk else None
except: gemma_client = None

# Modelos disponiveis
MODELS = {
    'claude': {
        'nome':    'Anthropic Claude Sonnet',
        'id':      'claude-sonnet-4-6',
        'cor':     '#1e3a5f',
        'api':     'Anthropic API',
        'custo_input':  3.0,
        'custo_output': 15.0,
        'gratis':  False,
        'badge':   'Parte 2',
    },
    'gemma4': {
        'nome':    'Google Gemma 4 31B',
        'id':      'gemma-4-31b-it',
        'cor':     '#2e6fad',
        'api':     'Google AI Studio',
        'custo_input':  0.0,
        'custo_output': 0.0,
        'gratis':  True,
        'badge':   'Novo — Abr 2026',
    },
    'gemma3': {
        'nome':    'Google Gemma 3 27B',
        'id':      'gemma-3-27b-it',
        'cor':     '#4a90c4',
        'api':     'Google AI Studio',
        'custo_input':  0.0,
        'custo_output': 0.0,
        'gratis':  True,
        'badge':   'Parte 1',
    },
}

def get_model_config():
    sel = st.session_state.get('modelo_sel', 'claude')
    if sel not in MODELS: sel = 'claude'
    return MODELS[sel]

def is_gemma(cfg=None):
    cfg = cfg or get_model_config()
    return cfg['id'].startswith('gemma')


def gemma_generate(prompt, model_id):
    """Chama a API Gemma com fallback automatico de modelos."""
    ids = [model_id]
    if 'gemma-4' in model_id:
        ids = ['gemma-4-31b-it','gemma-4-26b-a4b-it','gemma-4-e4b-it']
    for mid in ids:
        try:
            r = gemma_client.models.generate_content(model=mid, contents=prompt)
            return r.text, r
        except Exception as e:
            last = str(e)
            continue
    raise Exception(f"Nenhum modelo Gemma disponivel. Ultimo erro: {last}")

def llm_call(prompt, max_t=500):
    cfg = get_model_config()
    if cfg['id'].startswith('claude'):
        if not claude_client: return 'API Key Claude nao configurada.', 0, 0
        r = claude_client.messages.create(
            model=cfg['id'], max_tokens=max_t,
            messages=[{"role":"user","content":prompt}])
        return r.content[0].text.strip(), r.usage.input_tokens, r.usage.output_tokens
    else:
        # Gemma 3 e Gemma 4 — mesma API Google
        if not gemma_client: return 'API Key Gemma nao configurada.', 0, 0
        try:
            r   = gemma_client.models.generate_content(model=cfg['id'], contents=prompt)
            txt = r.text
        except Exception as e:
            txt = f'Erro Gemma: {e}'
        ti = int(len(prompt.split())*1.3)
        to = int(len(txt.split())*1.3)
        return txt, ti, to

# Compatibilidade com funcoes existentes
def get_client():
    cfg = get_model_config()
    return claude_client if cfg['id'].startswith('claude') else gemma_client

MODEL = 'claude-sonnet-4-6'  # fallback para compatibilidade

# ── Banco ──────────────────────────────────────────────────────────────────
DB_PATH = next((p for p in ['olist.db','./data/olist.db',
                '/content/drive/MyDrive/Agente de IA/olist.db'] if os.path.exists(p)), None)
SCHEMA  = """
SQLite — Olist E-Commerce BR (2016-2018).
orders(order_id,customer_id,order_status,order_purchase_timestamp,
       order_delivered_customer_date,order_estimated_delivery_date)
customers(customer_id,customer_unique_id,customer_city,customer_state)
items(order_id,product_id,seller_id,price,freight_value)
payments(order_id,payment_type,payment_installments,payment_value)
reviews(review_id,order_id,review_score,review_comment_message)
products(product_id,product_category_name,product_category_name_english)
sellers(seller_id,seller_city,seller_state)
REGRAS: atraso=delivered>estimated. entregues=status='delivered'. Use LIMIT."""

# ── Guardrails ─────────────────────────────────────────────────────────────
TOP  = ['pedido','cliente','produto','venda','pagamento','entrega','frete','avalia',
        'nota','categoria','estado','cidade','vendedor','receita','valor','preco',
        'atraso','status','ticket','compra','total','media','quantos','quanto']
BLOQ = ['cpf','rg','senha','password','dados pessoais','nome completo']

def validar(p):
    pl = p.lower()
    if len(p.strip())<5:          return {'ok':False,'msg':'Pergunta muito curta.'}
    if len(p)>500:                return {'ok':False,'msg':'Pergunta muito longa.'}
    if any(b in pl for b in BLOQ): return {'ok':False,'msg':'Nao forneco dados pessoais.'}
    if not any(t in pl for t in TOP): return {'ok':False,'msg':'So respondo sobre e-commerce.'}
    return {'ok':True,'msg':''}

# ── Text-to-SQL ────────────────────────────────────────────────────────────
def llm(prompt, max_t=500):
    return llm_call(prompt, max_t)

def sql_exec(sql):
    try:
        c=sqlite3.connect(DB_PATH); df=pd.read_sql_query(sql,c); c.close(); return df,None
    except Exception as e: return None,str(e)

def gerar_sql(perg): 
    txt,_,_=llm(f"Especialista SQL SQLite. APENAS SQL, sem markdown.\nSchema:{SCHEMA}\nPergunta:{perg}\nSQL:")
    return txt.replace('```sql','').replace('```','').strip()

def corrigir_sql(perg,sql_e,err):
    txt,_,_=llm(f"Corrija o SQL. APENAS SQL corrigido.\nSchema:{SCHEMA}\nPergunta:{perg}\nSQL:{sql_e}\nErro:{err}\nSQL corrigido:")
    return txt.replace('```sql','').replace('```','').strip()

def sql_retry(perg,max_t=2):
    sql=gerar_sql(perg); df,err=sql_exec(sql); t=1; corr=False
    while err and t<max_t:
        sql=corrigir_sql(perg,sql,err); df,err=sql_exec(sql); t+=1; corr=True
    return df,sql,t,corr

def interpretar(perg,df):
    txt,ti,to=llm(f"Analista e-commerce BR. Interprete em portugues. Max 3 linhas. Sem mencionar SQL.\nPergunta:{perg}\nResultado:{df.to_string(index=False)}\nResposta:")
    return txt,ti,to

# ── Avaliador ──────────────────────────────────────────────────────────────
PERGS_N=['ticket','valor','preco','media','nota','taxa','total','quantos','frete']

def extr_num(t):
    t=t.replace(',','.').replace('R$','').replace('%','')
    ns=re.findall(r'\d+\.?\d*',t); return float(ns[0]) if ns else None

def aval_num(resp,gt):
    vg=extr_num(gt);vr=extr_num(resp)
    if not vg or not vr: return {'status':'sem_numero','score':None,'metodo':'numerico'}
    e=abs(vg-vr)/vg if vg else 0
    if e<=.02: return {'status':'correta',   'score':1.0,'metodo':'numerico'}
    if e<=.10: return {'status':'parcial',   'score':.5,'metodo':'numerico'}
    return             {'status':'alucinacao','score':0.0,'metodo':'numerico'}

def aval_llm(perg,resp,gt):
    try:
        txt,_,_=llm(f'Avalie. Retorne APENAS JSON:\n{{"score":1.0,"status":"correta","justificativa":"texto"}}\nscore:1=correta,.5=parcial,0=incorreta\nPergunta:{perg}\nGT:{gt}\nResposta:{resp}\nJSON:',150)
        m=re.search(r'\{.*\}',txt,re.DOTALL)
        if m: r=json.loads(m.group()); r['metodo']='llm_juiz'; return r
    except: pass
    return {'status':'sem_ground_truth','score':None,'metodo':'llm_juiz'}

def avaliar(perg,resp,gt):
    if not gt: return {'status':'sem_ground_truth','score':None,'metodo':'nenhum'}
    if any(k in perg.lower() for k in PERGS_N):
        r=aval_num(resp,gt)
        if r['status']!='sem_numero': return r
    return aval_llm(perg,resp,gt)

def custo(ti,to):
    cfg = get_model_config()
    c   = (ti/1e6*cfg['custo_input']) + (to/1e6*cfg['custo_output'])
    return {'tokens_input':ti,'tokens_output':to,'total_tokens':ti+to,
            'custo_usd':round(c,6),'modelo':cfg['nome']}

# ── Ground Truth ───────────────────────────────────────────────────────────
def gt(p):
    if not DB_PATH: return None
    pl=p.lower(); c=sqlite3.connect(DB_PATH)
    try:
        if any(x in pl for x in ['mais pedidos','mais compras']):
            df=pd.read_sql("SELECT customer_state,COUNT(*) t FROM orders o JOIN customers c ON o.customer_id=c.customer_id GROUP BY customer_state ORDER BY t DESC LIMIT 1",c)
            return f"{df.iloc[0]['customer_state']} com {df.iloc[0]['t']:,} pedidos"
        if any(x in pl for x in ['ticket medio','valor medio','preco medio']):
            df=pd.read_sql("SELECT ROUND(AVG(price),2) v FROM items",c); return f"R$ {df.iloc[0]['v']}"
        if any(x in pl for x in ['nota media','avaliacao media']):
            df=pd.read_sql("SELECT ROUND(AVG(review_score),2) v FROM reviews",c); return f"{df.iloc[0]['v']} de 5.0"
        if any(x in pl for x in ['taxa de atraso','atraso']):
            if any(x in pl for x in ['maior','mais','estado']):
                df=pd.read_sql("SELECT customer_state,ROUND(AVG(CASE WHEN order_delivered_customer_date>order_estimated_delivery_date THEN 1.0 ELSE 0.0 END)*100,2) t FROM orders o JOIN customers c ON o.customer_id=c.customer_id WHERE order_status='delivered' GROUP BY customer_state ORDER BY t DESC LIMIT 1",c)
                return f"{df.iloc[0]['customer_state']} com {df.iloc[0]['t']}%"
            df=pd.read_sql("SELECT ROUND(AVG(CASE WHEN order_delivered_customer_date>order_estimated_delivery_date THEN 1.0 ELSE 0.0 END)*100,1) t FROM orders WHERE order_status='delivered'",c)
            return f"{df.iloc[0]['t']}% dos pedidos"
        if any(x in pl for x in ['total de pedidos','quantos pedidos']):
            df=pd.read_sql("SELECT COUNT(*) t FROM orders",c); return f"{df.iloc[0]['t']:,} pedidos"
        if any(x in pl for x in ['pagamento','forma de pagamento']):
            o='ASC' if any(x in pl for x in ['menos','menor','raro']) else 'DESC'
            df=pd.read_sql(f"SELECT payment_type,COUNT(*) t FROM payments GROUP BY payment_type ORDER BY t {o} LIMIT 1",c)
            return f"{df.iloc[0]['payment_type']} com {df.iloc[0]['t']:,} transacoes"
        if any(x in pl for x in ['categoria','mais vendida','maior receita']):
            o='ASC' if any(x in pl for x in ['menos','menor']) else 'DESC'
            df=pd.read_sql(f"SELECT product_category_name_english,ROUND(SUM(price),2) r FROM items i JOIN products p ON i.product_id=p.product_id GROUP BY product_category_name_english ORDER BY r {o} LIMIT 1",c)
            return f"{df.iloc[0]['product_category_name_english']} R$ {df.iloc[0]['r']:,.2f}"
        if any(x in pl for x in ['frete medio','valor do frete']):
            df=pd.read_sql("SELECT ROUND(AVG(freight_value),2) v FROM items",c); return f"R$ {df.iloc[0]['v']}"
    except: pass
    finally: c.close()
    return None

# ── Responder ──────────────────────────────────────────────────────────────
def responder(perg):
    t0=time.time(); val=validar(perg)
    if not val['ok']:
        return {'pergunta':perg,'resposta':val['msg'],'guardrail':True,
                'gt_val':None,'aval':None,'custo':None,'latencia':int((time.time()-t0)*1000)}
    try:
        df,sql,tent,corr=sql_retry(perg)
        if df is not None and not df.empty: txt,ti,to=interpretar(perg,df)
        else: txt,ti,to='Nao encontrei informacoes suficientes.',0,0
    except Exception as e: txt,ti,to=f'Erro:{e}',0,0
    gv=gt(perg)
    return {'pergunta':perg,'resposta':txt,'guardrail':False,'gt_val':gv,
            'aval':avaliar(perg,txt,gv),'custo':custo(ti,to),'latencia':int((time.time()-t0)*1000)}

# ── Insights ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def gerar_insights(_hist_key, _modelo_key='claude'):
    if not DB_PATH or not (claude_client or gemma_client): return None
    c=sqlite3.connect(DB_PATH)
    try:
        top_est=pd.read_sql("SELECT customer_state,COUNT(*) t FROM orders o JOIN customers c ON o.customer_id=c.customer_id GROUP BY customer_state ORDER BY t DESC LIMIT 5",c).to_string(index=False)
        top_cat=pd.read_sql("SELECT product_category_name_english,ROUND(SUM(price),2) r FROM items i JOIN products p ON i.product_id=p.product_id GROUP BY product_category_name_english ORDER BY r DESC LIMIT 5",c).to_string(index=False)
        pgtos=pd.read_sql("SELECT payment_type,COUNT(*) t FROM payments GROUP BY payment_type ORDER BY t DESC",c).to_string(index=False)
        avals=pd.read_sql("SELECT review_score,COUNT(*) t FROM reviews GROUP BY review_score ORDER BY review_score",c).to_string(index=False)
        atrasos=pd.read_sql("SELECT customer_state,ROUND(AVG(CASE WHEN order_delivered_customer_date>order_estimated_delivery_date THEN 1.0 ELSE 0.0 END)*100,1) t FROM orders o JOIN customers c ON o.customer_id=c.customer_id WHERE order_status='delivered' GROUP BY customer_state ORDER BY t DESC LIMIT 5",c).to_string(index=False)
    finally: c.close()
    txt,ti,to=llm(f"Analista senior e-commerce BR. 5 insights estrategicos, uma linha cada, comecando com -.\nDados: Estados:{top_est}|Categorias:{top_cat}|Pagamentos:{pgtos}|Avaliacoes:{avals}|Atrasos:{atrasos}\n5 INSIGHTS:",800)
    return {'insights':txt,'tokens':ti+to,'custo_usd':round((ti/1e6*3)+(to/1e6*15),6)}

# ── Benchmark data ─────────────────────────────────────────────────────────
# Atualize com seus resultados reais dos notebooks
BM = {
    'perguntas': ['Ticket medio?','Nota media?','Taxa atraso?','Total pedidos?',
                  'Frete medio?','Mais pedidos?','Pagamento mais usado?','Categoria top?'],
    'gemma3': {
        'modelo':   'Google Gemma 3 27B','cor':A5,
        'acuracia': [1,0,1,1,1,1,1,0],
        'latencias':[1648,1415,1782,1791,1350,1620,1480,1900],
        # Tokens: estimativa via contagem de palavras (Gemma nao retorna usage real)
        'tokens_input':  [312,298,335,280,290,320,305,340],
        'tokens_output': [85, 72, 91, 68, 78, 88, 82, 95],
        'custo':0.0,'avaliacao':'Overlap de palavras','dados':'CSV + RAG + FAISS',
        'badge':'Parte 1',
    },
    'gemma4': {
        'modelo':   'Google Gemma 4 27B','cor':A4,
        'acuracia': [1,1,1,1,1,1,1,1],
        'latencias':[1200,1100,1400,1000,1300,1150,1250,1350],
        # Gemma 4 — contexto maior, prompts mais eficientes
        'tokens_input':  [290,275,310,265,275,300,285,315],
        'tokens_output': [78, 65, 84, 61, 72, 81, 75, 88],
        'custo':0.0,'avaliacao':'Overlap de palavras','dados':'CSV + RAG + FAISS',
        'badge':'Novo — Abril 2026',
    },
    'claude': {
        'modelo':   'Anthropic Claude Sonnet','cor':A2,
        'acuracia': [1,1,0,1,1,1,1,1],
        'latencias':[8547,7200,9100,6800,7500,8200,7800,9300],
        # Claude: tokens reais da API (usage.input_tokens + usage.output_tokens)
        # Maior consumo porque inclui: SQL gerado + interpretacao + avaliacao LLM juiz
        'tokens_input':  [185,192,201,178,188,195,183,207],
        'tokens_output': [142,138,157,129,145,151,140,163],
        'custo':0.001706,'avaliacao':'Numerico + LLM Juiz','dados':'SQLite + Text-to-SQL',
        'badge':'Parte 2',
    },
}

# ══════════════════════════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════════════════════════
# Inicializar modelo selecionado
if 'modelo_sel' not in st.session_state:
    st.session_state['modelo_sel'] = 'claude'

with st.sidebar:
    st.markdown("### Modelo")

    LABELS = {
        'claude':  'Claude Sonnet',
        'gemma4':  'Gemma 4 31B',
        'gemma3':  'Gemma 3 27B',
    }
    modelo_opcao = st.radio(
        "Selecione o modelo:",
        options=['claude', 'gemma4', 'gemma3'],
        format_func=lambda x: LABELS[x],
        index=['claude','gemma4','gemma3'].index(st.session_state.get('modelo_sel','claude')),
        key='modelo_radio'
    )

    if modelo_opcao != st.session_state['modelo_sel']:
        st.session_state['modelo_sel'] = modelo_opcao
        st.session_state['hist'] = []
        st.rerun()



    cfg = get_model_config()
    st.markdown(f"""
    <div style="background:{cfg['cor']}15;border:1px solid {cfg['cor']}40;
                border-radius:6px;padding:10px 12px;margin-top:8px;font-size:.78rem">
        <strong style="color:{cfg['cor']}">{cfg['nome']}</strong><br>
        <span style="color:#888">API: {cfg['api']}</span><br>
        <span style="color:#888">Custo: {'Gratuito' if cfg['gratis'] else '$3/1M tokens'}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Status das conexoes
    st.markdown("**Status:**")
    if claude_client: st.success("Claude conectado")
    else: st.error("ANTHROPIC_API_KEY ausente")
    if gemma_client: st.success("Gemma conectado")
    else: st.warning("GEMINI_API_KEY ausente")
    if DB_PATH: st.success("Banco SQLite OK")
    else: st.error("olist.db nao encontrado")

    st.divider()
    st.markdown("**Exemplos:**")
    for ex in ["Qual estado tem mais pedidos?","Qual e o ticket medio?","Qual pagamento mais usado?",
               "Qual pagamento menos usado?","Qual categoria gera mais receita?",
               "Qual a taxa de atraso?","Qual a nota media?","Quantos pedidos no total?"]:
        if st.button(ex, key=ex, use_container_width=True): st.session_state['q']=ex
    st.divider()
    st.markdown(f"<div style='font-size:.72rem;color:#888;text-align:center'>Rafael Reghine Munhoz<br>Data Analyst | MBA USP<br><a href='https://github.com/rreghine' style='color:{A3}'>github.com/rreghine</a></div>",unsafe_allow_html=True)

cfg_h = get_model_config()
st.markdown(f"""<div class="header-wrap">
<div><div class="header-title">E-Commerce AI Agent<span class="header-badge">Multi-LLM</span></div>
<div class="header-sub">{cfg_h['nome']} &nbsp;·&nbsp; Text-to-SQL &nbsp;·&nbsp; Guardrails &nbsp;·&nbsp; Ground Truth Hibrido &nbsp;·&nbsp; LLM como Juiz &nbsp;·&nbsp; MLflow</div></div>
<div class="header-right"><a href="https://github.com/rreghine/ai-agent-part2" style="color:rgba(255,255,255,.7);text-decoration:none">github.com/rreghine</a><br>Brazilian E-Commerce · 99.441 pedidos</div></div>""",unsafe_allow_html=True)

if 'hist' not in st.session_state:        st.session_state['hist']=[]
if 'bm_resultados' not in st.session_state:   st.session_state['bm_resultados']={}
if 'bm_modelos' not in st.session_state:      st.session_state['bm_modelos']=[]
if 'bm_perguntas' not in st.session_state:    st.session_state['bm_perguntas']=[]
if 'insights_todos' not in st.session_state:  st.session_state['insights_todos']={}

tab1,tab2,tab3,tab4 = st.tabs(["  Agente  ","  Metricas  ","  Benchmark  ","  Dashboard e Insights  "])

# ── ABA 1 — AGENTE ─────────────────────────────────────────────────────────
with tab1:
    # Badge do modelo ativo
    cfg_t1 = get_model_config()
    st.markdown(f"""
    <div style="background:{cfg_t1['cor']}10;border:1px solid {cfg_t1['cor']}30;
                border-radius:5px;padding:8px 14px;margin-bottom:12px;
                display:flex;align-items:center;gap:10px">
        <span style="font-size:.72rem;font-weight:700;color:{cfg_t1['cor']};text-transform:uppercase;letter-spacing:.5px">
            Modelo ativo:
        </span>
        <span style="font-size:.82rem;font-weight:600;color:#2c3e50">{cfg_t1['nome']}</span>
        <span style="margin-left:auto;font-size:.7rem;color:#888">
            {'Gratuito' if cfg_t1['gratis'] else '$3/1M tokens input · $15/1M tokens output'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    default=st.session_state.pop('q','')
    c1,c2=st.columns([5,1])
    with c1: perg=st.text_input("p",value=default,placeholder="Ex: Qual estado tem mais pedidos?",label_visibility="collapsed")
    with c2: send=st.button("Enviar",type="primary",use_container_width=True)

    if send and perg.strip():
        cfg_c = get_model_config()
        if not DB_PATH:
            st.error("Banco nao encontrado.")
        else:
            cliente_ok = (claude_client if cfg_c["id"].startswith("claude") else gemma_client) is not None
            if not cliente_ok:
                api_k = "ANTHROPIC_API_KEY" if cfg_c["id"].startswith("claude") else "GEMINI_API_KEY"
                st.error(f"Configure {api_k} nos Secrets.")
            else:
                with st.spinner("Consultando..."):
                    res=responder(perg); st.session_state["hist"].insert(0,res)
        st.markdown('<div class="sec-label">Historico de Conversas</div>',unsafe_allow_html=True)
        for i,r in enumerate(st.session_state['hist']):
            with st.expander(f"**{r['pergunta']}**",expanded=(i==0)):
                if r.get('guardrail'): st.markdown(f'<div class="guard-box">{r["resposta"]}</div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="resp-box">{r["resposta"]}</div>',unsafe_allow_html=True)
                    av=r.get('aval',{});c=r.get('custo') or {}
                    sm={'correta':f'<span style="color:{A3}">Correta</span>','parcial':f'<span style="color:{AM}">Parcial</span>','alucinacao':f'<span style="color:{VR}">Alucinacao</span>','sem_ground_truth':'Sem GT'}
                    just=av.get('justificativa','')
                    st.markdown(f"""<div class="eval-row">
<div class="eval-cell"><div class="eval-lbl">Ground Truth</div><div class="eval-val">{r.get('gt_val') or '—'}</div></div>
<div class="eval-cell"><div class="eval-lbl">Avaliacao</div><div class="eval-val">{sm.get(av.get('status',''),'—')}</div></div>
<div class="eval-cell"><div class="eval-lbl">Metodo</div><div class="eval-val">{av.get('metodo','—')}</div></div>
<div class="eval-cell"><div class="eval-lbl">Tokens</div><div class="eval-val">{c.get('total_tokens',0):,}</div></div>
<div class="eval-cell"><div class="eval-lbl">Latencia</div><div class="eval-val">{r.get('latencia',0)}ms</div></div>
<div class="eval-cell"><div class="eval-lbl">Custo</div><div class="eval-val">${c.get('custo_usd',0):.6f}</div></div>
</div>{"<div style='font-size:.78rem;color:#888;margin-top:8px;padding:6px 10px;background:#f8f9fa;border-radius:4px'><strong>Justificativa:</strong> "+just+"</div>" if just else ""}""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("Limpar historico"): st.session_state['hist']=[]; st.rerun()

# ── ABA 2 — METRICAS ───────────────────────────────────────────────────────
with tab2:
    if not st.session_state['hist']: st.info("Faca perguntas na aba Agente.")
    else:
        validas=[r for r in st.session_state['hist'] if not r.get('guardrail')]
        todos=st.session_state['hist']
        total=len(todos); corr=sum(1 for r in validas if r.get('aval') and r['aval']['status']=='correta')
        guards=sum(1 for r in todos if r.get('guardrail'))
        tok_t=sum(r['custo']['total_tokens'] for r in validas if r.get('custo'))
        lat_m=np.mean([r['latencia'] for r in todos]) if todos else 0
        cg4=sum(r['custo']['custo_usd'] for r in validas if r.get('custo'))
        com_gt=[r for r in validas if r.get('aval') and r['aval']['status'] in ['correta','parcial','alucinacao']]
        t_hal=(sum(1 for r in com_gt if r['aval']['status']=='alucinacao')/len(com_gt)*100) if com_gt else 0
        n_num=sum(1 for r in validas if r.get('aval') and r['aval'].get('metodo')=='numerico')
        n_llm=sum(1 for r in validas if r.get('aval') and r['aval'].get('metodo')=='llm_juiz')

        cfg_m = get_model_config()
        custo_label = "Gratuito" if cfg_m['gratis'] else f"${cg4:.5f}"
        kpis=[("Total Queries",str(total),A3,"sessao atual"),("Corretas",str(corr),A2,"com ground truth"),
              ("Alucinacao",f"{t_hal:.1f}%",VR,"vs ground truth"),("Guardrails",str(guards),A4,"ativados"),
              ("Tokens Reais",f"{tok_t:,}",A3,"uso real API"),("Custo Modelo",custo_label,AM,"custo real")]
        cols=st.columns(6)
        for col,(lbl,val,cor,sub) in zip(cols,kpis):
            with col:
                st.markdown(f'<div class="kpi-card" style="--c:{cor}"><div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        for col,lbl,val,cor,sub in [(c1,"Aval. Numericas",str(n_num),A3,"comparacao matematica"),
                                     (c2,"Aval. LLM Juiz",str(n_llm),A2,"Claude avalia Claude"),
                                     (c3,"Latencia Media",f"{lat_m:.0f}ms",A4,"por query")]:
            with col: st.markdown(f'<div class="kpi-card" style="--c:{cor}"><div class="kpi-label">{lbl}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

        if len(validas)>=2:
            cmap=LinearSegmentedColormap.from_list('b',[A6,A4,A3,A2,A1])
            amap={'correta':'Correta','parcial':'Parcial','alucinacao':'Alucinacao','sem_ground_truth':'Sem GT','guardrail':'Guardrail'}
            acor={'correta':A3,'parcial':A4,'alucinacao':VR,'sem_ground_truth':A6,'guardrail':AM}
            plt.rcParams.update({'font.family':'DejaVu Sans','axes.facecolor':WH,'figure.facecolor':WH,'axes.grid':True,'grid.color':'#edf2f7','grid.linewidth':.7,'axes.axisbelow':True})
            fig=plt.figure(figsize=(16,10),facecolor=WH)
            gs=gridspec.GridSpec(2,3,figure=fig,hspace=.65,wspace=.4,top=.88,bottom=.10,left=.07,right=.97)
            fig.text(.5,.94,'Performance Dashboard — Claude Edition',ha='center',fontsize=13,fontweight='bold',color=A1)
            fig.text(.5,.915,'Anthropic Claude  ·  Text-to-SQL  ·  Ground Truth Hibrido  ·  LLM como Juiz  ·  MLflow',ha='center',fontsize=9,color=TL,style='italic')
            fig.add_artist(plt.Line2D([.07,.97],[.904,.904],transform=fig.transFigure,color=BD,lw=1))

            def sp_off(ax):
                for s in ['top','right']: ax.spines[s].set_visible(False)
                ax.spines['left'].set_color(BD); ax.spines['bottom'].set_color(BD); ax.tick_params(colors=TL,labelsize=9)

            ax1=fig.add_subplot(gs[0,0])
            avs=pd.Series([r['aval']['status'] for r in validas if r.get('aval')]).value_counts()
            b1=ax1.barh([amap.get(k,k) for k in avs.index],avs.values,color=[acor.get(k,A3) for k in avs.index],edgecolor=WH,height=.5)
            for b,v in zip(b1,avs.values): ax1.text(b.get_width()+avs.max()*.03,b.get_y()+b.get_height()/2,str(v),va='center',fontsize=10,fontweight='700',color=A1)
            ax1.set_title('Avaliacao das Respostas',fontweight='bold',color=A1,fontsize=11,pad=10); ax1.set_xlim(0,avs.max()*1.35); sp_off(ax1)

            ax2=fig.add_subplot(gs[0,1])
            mets=pd.Series([r['aval'].get('metodo','') for r in validas if r.get('aval')]).value_counts()
            mm={'numerico':'Numerico','llm_juiz':'LLM Juiz','nenhum':'Sem GT'}; mc={'numerico':A3,'llm_juiz':A2,'nenhum':A6,'':AM}
            b2=ax2.bar([mm.get(k,k) for k in mets.index],mets.values,color=[mc.get(k,A4) for k in mets.index],edgecolor=WH,width=.5)
            for b,v in zip(b2,mets.values): ax2.text(b.get_x()+b.get_width()/2,b.get_height()+mets.max()*.03,str(v),ha='center',va='bottom',fontsize=11,fontweight='700',color=A1)
            ax2.set_title('Metodo de Avaliacao',fontweight='bold',color=A1,fontsize=11,pad=10); ax2.set_ylim(0,mets.max()*1.35); sp_off(ax2)

            ax3=fig.add_subplot(gs[0,2])
            lats=[r['latencia'] for r in todos]
            ax3.fill_between(range(len(lats)),lats,alpha=.12,color=A4)
            ax3.plot(range(len(lats)),lats,color=A3,lw=2.5,marker='o',ms=6,mfc=WH,mec=A3,mew=2)
            for x,y in enumerate(lats): ax3.text(x,y+max(lats)*.05,f'{int(y)}ms',ha='center',va='bottom',fontsize=8,fontweight='600',color=A1)
            ax3.axhline(np.mean(lats),color=VR,lw=1.5,ls='--',label=f'Media:{np.mean(lats):.0f}ms'); ax3.legend(fontsize=8); ax3.set_ylim(0,max(lats)*1.35)
            ax3.set_title('Latencia por Query (ms)',fontweight='bold',color=A1,fontsize=11,pad=10); sp_off(ax3)

            ax4=fig.add_subplot(gs[1,0])
            dfv=pd.DataFrame([{'ti':r['custo']['tokens_input'],'to':r['custo']['tokens_output']} for r in validas if r.get('custo')]).reset_index(drop=True)
            xp=np.arange(len(dfv));w=.35
            ax4.bar(xp-w/2,dfv['ti'],width=w,label='Input',color=A3,edgecolor=WH); ax4.bar(xp+w/2,dfv['to'],width=w,label='Output',color=A4,edgecolor=WH)
            ax4.legend(fontsize=8); ax4.set_title('Tokens Input vs Output',fontweight='bold',color=A1,fontsize=11,pad=10); sp_off(ax4)

            ax5=fig.add_subplot(gs[1,1])
            sc=pd.Series([r['aval']['status'] for r in validas if r.get('aval')]).value_counts()
            w5,t5,at5=ax5.pie(sc.values,labels=[f'{amap.get(k,k)}\n({v})' for k,v in sc.items()],colors=[acor.get(k,A3) for k in sc.index],autopct='%1.1f%%',startangle=90,wedgeprops=dict(width=.55,edgecolor=WH,linewidth=2),textprops=dict(fontsize=8,color=A1))
            for a in at5: a.set_fontsize(8); a.set_fontweight('bold'); a.set_color(WH)
            ax5.set_title('Distribuicao de Status',fontweight='bold',color=A1,fontsize=11,pad=10)

            ax6=fig.add_subplot(gs[1,2])
            cacum=np.cumsum([r['custo']['custo_usd'] for r in validas if r.get('custo')]); mc2=max(cacum.max(),.000001)
            b6=ax6.bar(range(len(cacum)),cacum,color=[cmap(v/mc2) for v in cacum],edgecolor=WH,width=.65)
            for b,v in zip(b6,cacum): ax6.text(b.get_x()+b.get_width()/2,b.get_height()+mc2*.03,f'${v:.5f}',ha='center',va='bottom',fontsize=7.5,fontweight='600',color=A1,rotation=35)
            ax6.set_ylim(0,mc2*1.35); ax6.set_title('Custo Acumulado (Claude)',fontweight='bold',color=A1,fontsize=11,pad=10); sp_off(ax6)

            fig.add_artist(plt.Line2D([.07,.97],[.058,.058],transform=fig.transFigure,color=BD,lw=1))
            fig.text(.5,.036,'Rafael Reghine Munhoz  ·  Data Analyst | MBA USP',ha='center',fontsize=8,fontweight='600',color=A2)
            fig.text(.5,.016,'github.com/rreghine  ·  linkedin.com/in/rafaelreghine',ha='center',fontsize=8,color=A4)
            st.pyplot(fig)
        else: st.info("Faca pelo menos 2 perguntas para ver os graficos.")

        st.markdown('<div class="sec-label">Historico Detalhado</div>',unsafe_allow_html=True)
        dfh=pd.DataFrame([{'Pergunta':r['pergunta'][:55]+'...' if len(r['pergunta'])>55 else r['pergunta'],
                            'Avaliacao':r['aval']['status'] if r.get('aval') else 'guardrail',
                            'Metodo':r['aval'].get('metodo','—') if r.get('aval') else '—',
                            'Tokens':r['custo']['total_tokens'] if r.get('custo') else 0,
                            'Latencia':f"{r['latencia']}ms",'Custo':f"${r['custo']['custo_usd']:.6f}" if r.get('custo') else '—'}
                           for r in st.session_state['hist']])
        st.dataframe(dfh,use_container_width=True,hide_index=True)

# ── ABA 3 — BENCHMARK AO VIVO ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-label">Benchmark ao Vivo — Mesma Pergunta · 3 Modelos · mesmo banco SQLite</div>',
                unsafe_allow_html=True)

    # Descricao do benchmark
    st.markdown(f"""
    <div style="background:{BR};border:1px solid {BD};border-radius:6px;
                padding:14px 18px;font-size:.84rem;color:{TX};margin-bottom:16px">
        <strong>Como funciona:</strong> as mesmas perguntas sao enviadas simultaneamente
        para os 3 modelos usando o mesmo banco SQLite e o mesmo metodo Text-to-SQL.
        A comparacao e 100% justa — unica variavel e o modelo de linguagem.
    </div>
    """, unsafe_allow_html=True)

    # Perguntas do benchmark
    PERGS_BM_DEFAULT = [
        'Qual e o ticket medio dos pedidos?',
        'Qual a nota media de avaliacao dos clientes?',
        'Qual e a taxa de atraso nas entregas?',
        'Quantos pedidos existem no total?',
        'Qual o valor medio do frete?',
        'Qual estado tem mais pedidos?',
        'Qual a forma de pagamento mais usada?',
        'Qual categoria gera mais receita?',
    ]

    # Selecionar quais modelos rodar
    st.markdown("**Selecione os modelos:**")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: usar_g3 = st.checkbox("Gemma 3 27B", value=True)
    with col_m2: usar_g4 = st.checkbox("Gemma 4 31B", value=True)
    with col_m3: usar_cl = st.checkbox("Claude Sonnet", value=True)

    # Editar perguntas
    with st.expander("Editar perguntas do benchmark"):
        pergs_txt = st.text_area(
            "Uma pergunta por linha:",
            value="\n".join(PERGS_BM_DEFAULT),

            height=200
        )
        PERGS_BM = [p.strip() for p in pergs_txt.split("\n") if p.strip()]


    st.markdown("<br>", unsafe_allow_html=True)
    rodar = st.button("Rodar Benchmark", type="primary", use_container_width=False)

    if rodar:
        if not DB_PATH:
            st.error("Banco olist.db nao encontrado.")
        elif not (usar_g3 or usar_g4 or usar_cl):
            st.warning("Selecione pelo menos um modelo.")
        else:
            modelos_selecionados = []
            if usar_g3: modelos_selecionados.append(('gemma3','Gemma 3 27B', 'gemma-3-27b-it', A5, False))
            if usar_g4: modelos_selecionados.append(('gemma4','Gemma 4 31B', 'gemma-4-31b-it', A4, False))
            if usar_cl: modelos_selecionados.append(('claude','Claude Sonnet','claude-sonnet-4-6',A2, True))

            resultados_bm = {k: [] for k,_,_,_,_ in modelos_selecionados}
            progresso = st.progress(0)
            status_txt = st.empty()
            total_ops  = len(PERGS_BM) * len(modelos_selecionados)
            op_atual   = 0

            for perg in PERGS_BM:
                for key, nome, model_id, cor, is_claude in modelos_selecionados:
                    status_txt.markdown(f"**Rodando:** `{nome}` → *{perg[:50]}...*")
                    t0 = time.time()
                    try:
                        # Gerar SQL com o modelo especifico
                        prompt_sql = f"""Especialista SQL SQLite. APENAS SQL, sem markdown.
Schema:{SCHEMA}
Pergunta:{perg}
SQL:"""
                        if is_claude:
                            if not claude_client: raise Exception("Claude nao configurado")
                            r_sql = claude_client.messages.create(
                                model=model_id, max_tokens=400,
                                messages=[{"role":"user","content":prompt_sql}])
                            sql = r_sql.content[0].text.strip().replace('```sql','').replace('```','').strip()
                            ti_sql = r_sql.usage.input_tokens
                            to_sql = r_sql.usage.output_tokens
                        else:
                            if not gemma_client: raise Exception("Gemma nao configurado")
                            r_sql = gemma_client.models.generate_content(model=model_id, contents=prompt_sql)
                            sql = r_sql.text.strip().replace('```sql','').replace('```','').strip()
                            ti_sql = int(len(prompt_sql.split())*1.3)
                            to_sql = int(len(sql.split())*1.3)

                        # Executar SQL
                        df_r, err = sql_exec(sql)
                        if err or df_r is None or df_r.empty:
                            # Tentar corrigir
                            prompt_fix = (f"Corrija o SQL. APENAS SQL.\n"
                                         f"Schema:{SCHEMA}\nSQL:{sql}\nErro:{err}\nSQL corrigido:")



                            if is_claude:
                                r_fix = claude_client.messages.create(model=model_id, max_tokens=400,
                                    messages=[{"role":"user","content":prompt_fix}])
                                sql = r_fix.content[0].text.strip().replace('```sql','').replace('```','').strip()
                                ti_sql += r_fix.usage.input_tokens; to_sql += r_fix.usage.output_tokens
                            else:
                                txt_fix, _ = gemma_generate(prompt_fix, model_id)
                                sql = txt_fix.strip().replace('```sql','').replace('```','').strip()
                            df_r, err = sql_exec(sql)

                        # Interpretar resultado
                        if df_r is not None and not df_r.empty:
                            prompt_int = (f"Analista e-commerce BR. Interprete em portugues. Max 2 linhas.\n"
                                          f"Pergunta:{perg}\nResultado:{df_r.to_string(index=False)}\nResposta:")


                            if is_claude:
                                r_int = claude_client.messages.create(model=model_id, max_tokens=200,
                                    messages=[{"role":"user","content":prompt_int}])
                                txt = r_int.content[0].text.strip()
                                ti_sql += r_int.usage.input_tokens; to_sql += r_int.usage.output_tokens
                            else:
                                txt_int, _ = gemma_generate(prompt_int, model_id)
                                txt = txt_int.strip()
                                ti_sql += int(len(prompt_int.split())*1.3)
                                to_sql += int(len(txt.split())*1.3)
                        else:
                            txt = 'Nao encontrei dados suficientes.'

                        # Ground truth e avaliacao
                        gv  = gt(perg)
                        aval_r = avaliar(perg, txt, gv)
                        lat = int((time.time()-t0)*1000)

                        # Custo real
                        if is_claude:
                            custo_q = (ti_sql/1e6*3.0) + (to_sql/1e6*15.0)
                        else:
                            custo_q = 0.0

                        resultados_bm[key].append({
                            'pergunta':  perg,
                            'resposta':  txt,
                            'sql':       sql,
                            'gt':        gv,
                            'status':    aval_r.get('status','—'),
                            'metodo':    aval_r.get('metodo','—'),
                            'tokens_in': ti_sql,
                            'tokens_out':to_sql,
                            'tokens_tot':ti_sql+to_sql,
                            'custo_usd': round(custo_q,6),
                            'latencia':  lat,
                        })
                    except Exception as e:
                        resultados_bm[key].append({
                            'pergunta':perg,'resposta':f'Erro: {e}','sql':'',
                            'gt':None,'status':'erro','metodo':'—',
                            'tokens_in':0,'tokens_out':0,'tokens_tot':0,
                            'custo_usd':0,'latencia':0,
                        })

                    op_atual += 1
                    progresso.progress(op_atual / total_ops)
                    time.sleep(0.5)

            st.session_state['bm_resultados']  = resultados_bm
            st.session_state['bm_modelos']     = modelos_selecionados
            st.session_state['bm_perguntas']   = PERGS_BM
            status_txt.markdown("**Benchmark concluido!**")
            progresso.progress(1.0)

    # Exibir resultados
    if st.session_state.get('bm_resultados'):
        res     = st.session_state['bm_resultados']
        mods    = st.session_state['bm_modelos']
        pergs   = st.session_state['bm_perguntas']

        st.markdown('<div class="sec-label">Resultados do Benchmark</div>',unsafe_allow_html=True)

        # KPI cards por modelo
        cols_kpi = st.columns(len(mods))
        for col,(key,nome,mid,cor,ic) in zip(cols_kpi, mods):
            dados = res[key]
            n_tot   = len(dados)
            n_corr  = sum(1 for d in dados if d['status']=='correta')
            n_hal   = sum(1 for d in dados if d['status']=='alucinacao')
            tok_med = np.mean([d['tokens_tot'] for d in dados]) if dados else 0
            lat_med = np.mean([d['latencia'] for d in dados]) if dados else 0
            custo_t = sum(d['custo_usd'] for d in dados)
            with col:
                st.markdown(f"""
                <div class="bench-card" style="border-top:3px solid {cor}">
                    <div style="font-size:.75rem;font-weight:700;color:{cor};text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">{nome}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                        <div style="text-align:center;padding:8px;background:{BR};border-radius:4px">
                            <div style="font-size:1.4rem;font-weight:700;color:{cor}">{n_corr}/{n_tot}</div>
                            <div style="font-size:.65rem;color:{TL}">CORRETAS</div>
                        </div>
                        <div style="text-align:center;padding:8px;background:{BR};border-radius:4px">
                            <div style="font-size:1.4rem;font-weight:700;color:{VR}">{n_hal}</div>
                            <div style="font-size:.65rem;color:{TL}">ALUCINACOES</div>
                        </div>
                        <div style="text-align:center;padding:8px;background:{BR};border-radius:4px">
                            <div style="font-size:1.1rem;font-weight:700;color:{A2}">{tok_med:.0f}</div>
                            <div style="font-size:.65rem;color:{TL}">TOKENS/QUERY</div>
                        </div>
                        <div style="text-align:center;padding:8px;background:{BR};border-radius:4px">
                            <div style="font-size:1.1rem;font-weight:700;color:{AM}">{lat_med:.0f}ms</div>
                            <div style="font-size:.65rem;color:{TL}">LATENCIA</div>
                        </div>
                    </div>
                    <div style="margin-top:10px;text-align:center;font-size:.78rem;font-weight:700;color:{'#2e6fad' if custo_t==0 else AM}">
                        {'Gratuito' if custo_t==0 else f'Custo total: ${custo_t:.5f}'}
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Graficos comparativos ao vivo
        cmap_bm = LinearSegmentedColormap.from_list('b',[A6,A4,A3,A2,A1])
        fig_bm, axs_bm = plt.subplots(2, 3, figsize=(16, 10), facecolor=WH)
        fig_bm.suptitle('Benchmark ao Vivo — Text-to-SQL · Mesmo Banco · Mesmas Perguntas',
                         fontsize=12, fontweight='bold', color=A1, y=1.01)

        def sp_bm(ax):
            for s in ['top','right']: ax.spines[s].set_visible(False)
            ax.spines['left'].set_color(BD); ax.spines['bottom'].set_color(BD)
            ax.set_facecolor(WH); ax.tick_params(colors=TL, labelsize=8)

        aval_cor_bm = {'correta':A3,'parcial':A4,'alucinacao':VR,'sem_ground_truth':A6,'erro':'#bdc3c7','—':'#bdc3c7'}
        n_pergs = len(pergs)
        x_bm    = np.arange(n_pergs)
        w_bm    = 0.8 / len(mods)

        # G1 — Acuracia por pergunta
        for i,(key,nome,_,cor,_) in enumerate(mods):
            accs = [1 if d['status']=='correta' else 0 for d in res[key]]
            offset = (i - len(mods)/2 + 0.5) * w_bm
            axs_bm[0,0].bar(x_bm+offset, accs, width=w_bm*0.9,
                            label=nome.split()[1], color=cor, edgecolor=WH)
        axs_bm[0,0].set_xticks(x_bm)
        axs_bm[0,0].set_xticklabels([f'Q{i+1}' for i in range(n_pergs)], fontsize=8)
        axs_bm[0,0].set_yticks([0,1]); axs_bm[0,0].set_yticklabels(['Falhou','Correta'], fontsize=8)
        axs_bm[0,0].legend(fontsize=7); axs_bm[0,0].set_title('Acuracia por Query', fontweight='bold', color=A1, fontsize=11, pad=8)
        sp_bm(axs_bm[0,0])

        # G2 — Latencia por pergunta
        for i,(key,nome,_,cor,_) in enumerate(mods):
            lats = [d['latencia'] for d in res[key]]
            offset = (i - len(mods)/2 + 0.5) * w_bm
            axs_bm[0,1].bar(x_bm+offset, lats, width=w_bm*0.9,
                            label=nome.split()[1], color=cor, edgecolor=WH)
        axs_bm[0,1].set_xticks(x_bm)
        axs_bm[0,1].set_xticklabels([f'Q{i+1}' for i in range(n_pergs)], fontsize=8)
        axs_bm[0,1].set_ylabel('ms', fontsize=9, color=TL)
        axs_bm[0,1].legend(fontsize=7); axs_bm[0,1].set_title('Latencia por Query (ms)', fontweight='bold', color=A1, fontsize=11, pad=8)
        sp_bm(axs_bm[0,1])

        # G3 — Tokens totais por pergunta
        for i,(key,nome,_,cor,_) in enumerate(mods):
            toks = [d['tokens_tot'] for d in res[key]]
            offset = (i - len(mods)/2 + 0.5) * w_bm
            axs_bm[0,2].bar(x_bm+offset, toks, width=w_bm*0.9,
                            label=nome.split()[1], color=cor, edgecolor=WH)
        axs_bm[0,2].set_xticks(x_bm)
        axs_bm[0,2].set_xticklabels([f'Q{i+1}' for i in range(n_pergs)], fontsize=8)
        axs_bm[0,2].set_ylabel('Tokens', fontsize=9, color=TL)
        axs_bm[0,2].legend(fontsize=7); axs_bm[0,2].set_title('Tokens Totais por Query', fontweight='bold', color=A1, fontsize=11, pad=8)
        sp_bm(axs_bm[0,2])

        # G4 — Acuracia geral (barras)
        nomes_bm = [n.split()[1] for _,n,_,_,_ in mods]
        cors_bm  = [c for _,_,_,c,_ in mods]
        accs_ger = [sum(1 for d in res[k] if d['status']=='correta')/len(res[k])*100 for k,_,_,_,_ in mods]
        b_ag = axs_bm[1,0].bar(nomes_bm, accs_ger, color=cors_bm, edgecolor=WH, width=0.45)
        for b,v in zip(b_ag, accs_ger):
            axs_bm[1,0].text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.0f}%',
                             ha='center', va='bottom', fontsize=13, fontweight='700', color=A1)
        axs_bm[1,0].set_ylim(0,120); axs_bm[1,0].set_title('Acuracia Geral (%)', fontweight='bold', color=A1, fontsize=11, pad=8)
        sp_bm(axs_bm[1,0])

        # G5 — Tokens input vs output por modelo
        x_mod = np.arange(len(mods)); w_mod = 0.35
        ti_med = [np.mean([d['tokens_in'] for d in res[k]]) for k,_,_,_,_ in mods]
        to_med = [np.mean([d['tokens_out'] for d in res[k]]) for k,_,_,_,_ in mods]
        axs_bm[1,1].bar(x_mod-w_mod/2, ti_med, width=w_mod, label='Input',  color=A3, edgecolor=WH)
        axs_bm[1,1].bar(x_mod+w_mod/2, to_med, width=w_mod, label='Output', color=A4, edgecolor=WH)
        for x,v in enumerate(ti_med): axs_bm[1,1].text(x-w_mod/2, v+2, f'{v:.0f}', ha='center', va='bottom', fontsize=8, fontweight='600', color=A1)
        for x,v in enumerate(to_med): axs_bm[1,1].text(x+w_mod/2, v+2, f'{v:.0f}', ha='center', va='bottom', fontsize=8, fontweight='600', color=A1)
        axs_bm[1,1].set_xticks(x_mod); axs_bm[1,1].set_xticklabels(nomes_bm, fontsize=9)
        axs_bm[1,1].legend(fontsize=8); axs_bm[1,1].set_ylabel('Tokens', fontsize=9, color=TL)
        axs_bm[1,1].set_title('Tokens Input vs Output', fontweight='bold', color=A1, fontsize=11, pad=8)
        sp_bm(axs_bm[1,1])

        # G6 — Latencia media por modelo
        lat_med_bm = [np.mean([d['latencia'] for d in res[k]]) for k,_,_,_,_ in mods]
        b_lat = axs_bm[1,2].bar(nomes_bm, lat_med_bm, color=cors_bm, edgecolor=WH, width=0.45)
        for b,v in zip(b_lat, lat_med_bm):
            axs_bm[1,2].text(b.get_x()+b.get_width()/2, b.get_height()+10, f'{v:.0f}ms',
                             ha='center', va='bottom', fontsize=10, fontweight='700', color=A1)
        axs_bm[1,2].set_title('Latencia Media (ms)', fontweight='bold', color=A1, fontsize=11, pad=8)
        axs_bm[1,2].set_ylabel('ms', fontsize=9, color=TL)
        sp_bm(axs_bm[1,2])

        fig_bm.add_artist(plt.Line2D([0.04,0.97],[0.05,0.05], transform=fig_bm.transFigure, color=BD, lw=1))
        fig_bm.text(0.5, 0.03, 'Rafael Reghine Munhoz  ·  Data Analyst | MBA USP  ·  github.com/rreghine',
                    ha='center', fontsize=8, color=A4)
        plt.tight_layout()
        st.pyplot(fig_bm)

        # Tabela detalhada por pergunta
        st.markdown('<div class="sec-label">Respostas por Pergunta</div>', unsafe_allow_html=True)

        for i, perg in enumerate(pergs):
            with st.expander(f"Q{i+1} — {perg}"):
                cols_resp = st.columns(len(mods))
                for col,(key,nome,_,cor,_) in zip(cols_resp, mods):
                    d = res[key][i]
                    st_ico = {'correta':'Correta','parcial':'Parcial','alucinacao':'Alucinacao',
                              'sem_ground_truth':'Sem GT','erro':'Erro','—':'—'}
                    st_cor = {'correta':A3,'parcial':AM,'alucinacao':VR,'sem_ground_truth':A6,'erro':'#bdc3c7','—':TL}
                    with col:
                        st.markdown(f"""
                        <div style="border-left:3px solid {cor};padding:10px 14px;background:{BR};border-radius:0 6px 6px 0;margin-bottom:8px">
                            <div style="font-size:.7rem;font-weight:700;color:{cor};margin-bottom:6px">{nome}</div>
                            <div style="font-size:.82rem;color:{TX};margin-bottom:8px">{d['resposta'][:200]}</div>
                            <div style="font-size:.7rem;color:{st_cor.get(d['status'],TL)};font-weight:700">{st_ico.get(d['status'],'—')}</div>
                            <div style="font-size:.68rem;color:{TL};margin-top:4px">
                                {d['tokens_tot']:,} tokens · {d['latencia']}ms
                                {f" · ${d['custo_usd']:.5f}" if d['custo_usd']>0 else " · Gratuito"}
                            </div>
                        </div>""", unsafe_allow_html=True)

        # Tabela resumo final
        st.markdown('<div class="sec-label">Resumo Final do Benchmark</div>', unsafe_allow_html=True)
        df_resumo = pd.DataFrame([{
            'Modelo':        nome,
            'Corretas':      f"{sum(1 for d in res[k] if d['status']=='correta')}/{len(res[k])}",
            'Acuracia':      f"{sum(1 for d in res[k] if d['status']=='correta')/len(res[k])*100:.0f}%",
            'Tokens/Query':  f"{np.mean([d['tokens_tot'] for d in res[k]]):.0f}",
            'Latencia Media':f"{np.mean([d['latencia'] for d in res[k]]):.0f}ms",
            'Custo Total':   'Gratuito' if all(d['custo_usd']==0 for d in res[k]) else f"${sum(d['custo_usd'] for d in res[k]):.5f}",
        } for k,nome,_,_,_ in mods])
        st.dataframe(df_resumo, use_container_width=True, hide_index=True)

# ── ABA 4 — DASHBOARD E INSIGHTS ──────────────────────────────────────────
with tab4:
    if not DB_PATH:
        st.error("Banco nao encontrado.")
    elif not (claude_client or gemma_client):
        st.error("Configure ANTHROPIC_API_KEY ou GEMINI_API_KEY nos Secrets.")
    else:
        # Header
        st.markdown(f"""
        <div style="font-size:.85rem;color:{TL};margin-bottom:12px">
            Os 3 modelos analisam os mesmos dados do banco e geram insights independentes.
            Compare qualidade, profundidade e custo de cada LLM em tempo real.
        </div>""", unsafe_allow_html=True)

        col_desc, col_btn = st.columns([5,1])
        with col_btn:
            gerar = st.button("Gerar Insights", type="primary", use_container_width=True)

        if gerar:
            st.cache_data.clear()
            st.session_state['insights_todos'] = {}

        # Definir modelos a rodar
        MODELOS_INS = [
            {'key':'gemma3', 'nome':'Gemma 3 27B',  'id':'gemma-3-27b-it',  'cor':A5,  'gratis':True,  'client':'gemma'},
            {'key':'gemma4', 'nome':'Gemma 4 31B',  'id':'gemma-4-31b-it',  'cor':A4,  'gratis':True,  'client':'gemma'},
            {'key':'claude', 'nome':'Claude Sonnet', 'id':'claude-sonnet-4-6','cor':A2, 'gratis':False, 'client':'claude'},
        ]

        # Funcao para gerar insights de um modelo especifico
        def gerar_insight_modelo(mod_cfg, hist_len=0):
            """Gera insights para um modelo especifico."""
            conn = sqlite3.connect(DB_PATH)
            try:
                top_est = pd.read_sql("SELECT customer_state,COUNT(*) t FROM orders o JOIN customers c ON o.customer_id=c.customer_id GROUP BY customer_state ORDER BY t DESC LIMIT 5", conn).to_string(index=False)
                top_cat = pd.read_sql("SELECT product_category_name_english,ROUND(SUM(price),2) r FROM items i JOIN products p ON i.product_id=p.product_id GROUP BY product_category_name_english ORDER BY r DESC LIMIT 5", conn).to_string(index=False)
                pgtos   = pd.read_sql("SELECT payment_type,COUNT(*) t FROM payments GROUP BY payment_type ORDER BY t DESC", conn).to_string(index=False)
                avals   = pd.read_sql("SELECT review_score,COUNT(*) t FROM reviews GROUP BY review_score ORDER BY review_score", conn).to_string(index=False)
                atrasos = pd.read_sql("SELECT customer_state,ROUND(AVG(CASE WHEN order_delivered_customer_date>order_estimated_delivery_date THEN 1.0 ELSE 0.0 END)*100,1) t FROM orders o JOIN customers c ON o.customer_id=c.customer_id WHERE order_status='delivered' GROUP BY customer_state ORDER BY t DESC LIMIT 5", conn).to_string(index=False)
            finally:
                conn.close()

            prompt = f"""Analista senior de e-commerce brasileiro.
Gere exatamente 5 insights estrategicos sobre os dados abaixo.
Formato: uma linha por insight, comecando com -.
Use numeros reais. Foque em oportunidades e riscos de negocio.

DADOS:
Estados: {top_est}
Categorias: {top_cat}
Pagamentos: {pgtos}
Avaliacoes: {avals}
Atrasos: {atrasos}

5 INSIGHTS:"""

            t0 = time.time()
            try:
                if mod_cfg['client'] == 'claude':
                    if not claude_client: return None
                    r  = claude_client.messages.create(
                        model=mod_cfg['id'], max_tokens=600,
                        messages=[{"role":"user","content":prompt}])
                    txt = r.content[0].text.strip()
                    ti  = r.usage.input_tokens
                    to  = r.usage.output_tokens
                else:
                    if not gemma_client: return None
                    txt, _ = gemma_generate(prompt, mod_cfg['id'])
                    ti = int(len(prompt.split())*1.3)
                    to = int(len(txt.split())*1.3)

                custo_v = 0.0 if mod_cfg['gratis'] else round((ti/1e6*3)+(to/1e6*15),6)
                return {
                    'insights':   txt,
                    'tokens':     ti+to,
                    'custo_usd':  custo_v,
                    'latencia':   int((time.time()-t0)*1000),
                    'modelo':     mod_cfg['nome'],
                }
            except Exception as e:
                return {'erro': str(e), 'modelo': mod_cfg['nome']}

        # Botao gerar — rodar os 3 modelos
        if gerar:
            resultados = {}
            progress   = st.progress(0)
            status     = st.empty()
            for i, mod in enumerate(MODELOS_INS):
                status.markdown(f"**Gerando insights com {mod['nome']}...**")
                resultados[mod['key']] = gerar_insight_modelo(mod)
                progress.progress((i+1)/len(MODELOS_INS))
            st.session_state['insights_todos'] = resultados
            status.empty(); progress.empty()

        # Exibir resultados
        if st.session_state.get('insights_todos'):
            res = st.session_state['insights_todos']

            # ── KPI resumo ────────────────────────────────────────────────
            st.markdown('<div class="sec-label">Custo e Tokens por Modelo</div>',
                        unsafe_allow_html=True)
            kpi_cols = st.columns(3)
            for col, mod in zip(kpi_cols, MODELOS_INS):
                r = res.get(mod['key'], {})
                with col:
                    if r and not r.get('erro'):
                        st.markdown(f"""
                        <div class="kpi-card" style="--c:{mod['cor']}">
                            <div class="kpi-label">{mod['nome']}</div>
                            <div class="kpi-value" style="font-size:1.2rem">{r['tokens']:,} tok</div>
                            <div class="kpi-sub">{'Gratuito' if mod['gratis'] else f"${r['custo_usd']:.6f}"} &nbsp;·&nbsp; {r['latencia']}ms</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="kpi-card" style="--c:{VR}">
                            <div class="kpi-label">{mod['nome']}</div>
                            <div class="kpi-value" style="font-size:1rem">Erro</div>
                            <div class="kpi-sub">{r.get('erro','—')[:60]}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Insights em colunas ───────────────────────────────────────
            st.markdown('<div class="sec-label">Insights por Modelo — Comparativo</div>',
                        unsafe_allow_html=True)

            ins_cols = st.columns(3)
            CORES = [A3, A2, A4, VR, AM]

            for col, mod in zip(ins_cols, MODELOS_INS):
                r = res.get(mod['key'], {})
                with col:
                    # Header do modelo
                    st.markdown(f"""
                    <div style="background:{mod['cor']};color:white;border-radius:6px 6px 0 0;
                                padding:10px 14px;font-size:.8rem;font-weight:700;
                                letter-spacing:.3px;margin-bottom:0">
                        {mod['nome']}
                        <span style="float:right;font-size:.72rem;opacity:.8">
                            {'Gratuito' if mod['gratis'] else '$3/1M tokens'}
                        </span>
                    </div>""", unsafe_allow_html=True)

                    if r and not r.get('erro'):
                        linhas = [l.strip() for l in r['insights'].split('\n') if l.strip()]
                        for i, linha in enumerate(linhas[:5]):
                            txt = linha.lstrip('-').lstrip('0123456789.').strip()
                            cor = CORES[i % len(CORES)]
                            st.markdown(f"""
                            <div style="border-left:3px solid {cor};background:{BR};
                                        padding:10px 12px;margin-bottom:6px;
                                        font-size:.8rem;color:{TX};line-height:1.6;
                                        border-radius:0 4px 4px 0">
                                <div style="font-size:.58rem;color:{cor};font-weight:700;
                                            text-transform:uppercase;letter-spacing:1px;
                                            margin-bottom:3px">Insight {i+1}</div>
                                {txt}
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#fff5f5;border:1px solid #fcc;border-radius:0 0 6px 6px;
                                    padding:14px;font-size:.8rem;color:{VR}">
                            Erro: {r.get('erro','Modelo nao disponivel')[:150]}
                        </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Grafico comparativo de tokens e custo ─────────────────────
            st.markdown('<div class="sec-label">Comparativo de Consumo</div>',
                        unsafe_allow_html=True)

            dados_ok = [(mod, res[mod['key']]) for mod in MODELOS_INS
                        if res.get(mod['key']) and not res[mod['key']].get('erro')]

            if len(dados_ok) >= 2:
                fig_ins, axes_ins = plt.subplots(1, 3, figsize=(14, 4), facecolor=WH)
                fig_ins.suptitle('Consumo para Gerar 5 Insights — Comparativo',
                                 fontsize=11, fontweight='bold', color=A1)

                nomes = [m['nome'].split()[1] for m,_ in dados_ok]
                cors  = [m['cor'] for m,_ in dados_ok]
                toks  = [r['tokens'] for _,r in dados_ok]
                lats  = [r['latencia'] for _,r in dados_ok]
                csts  = [r['custo_usd'] for _,r in dados_ok]

                def sp_ins(ax):
                    for s in ['top','right']: ax.spines[s].set_visible(False)
                    ax.spines['left'].set_color(BD); ax.spines['bottom'].set_color(BD)
                    ax.set_facecolor(WH); ax.tick_params(colors=TL, labelsize=9)

                # G1 — Tokens
                b1 = axes_ins[0].bar(nomes, toks, color=cors, edgecolor=WH, width=.45)
                for b,v in zip(b1,toks):
                    axes_ins[0].text(b.get_x()+b.get_width()/2, b.get_height()+5,
                                     f'{v:,}', ha='center', va='bottom',
                                     fontsize=10, fontweight='700', color=A1)
                axes_ins[0].set_title('Tokens Consumidos', fontweight='bold',
                                      color=A1, fontsize=10, pad=8)
                axes_ins[0].set_ylabel('Tokens', fontsize=9, color=TL)
                axes_ins[0].set_ylim(0, max(toks)*1.35)
                sp_ins(axes_ins[0])

                # G2 — Latencia
                b2 = axes_ins[1].bar(nomes, lats, color=cors, edgecolor=WH, width=.45)
                for b,v in zip(b2,lats):
                    axes_ins[1].text(b.get_x()+b.get_width()/2, b.get_height()+50,
                                     f'{v:,}ms', ha='center', va='bottom',
                                     fontsize=10, fontweight='700', color=A1)
                axes_ins[1].set_title('Latencia (ms)', fontweight='bold',
                                      color=A1, fontsize=10, pad=8)
                axes_ins[1].set_ylabel('ms', fontsize=9, color=TL)
                axes_ins[1].set_ylim(0, max(lats)*1.35)
                sp_ins(axes_ins[1])

                # G3 — Custo
                cst_labels = ['Gratuito' if c==0 else f'${c:.5f}' for c in csts]
                b3 = axes_ins[2].bar(nomes, csts, color=cors, edgecolor=WH, width=.45)
                for b,v,lbl in zip(b3,csts,cst_labels):
                    axes_ins[2].text(b.get_x()+b.get_width()/2,
                                     b.get_height()+max(csts)*0.02 if max(csts)>0 else 0.000001,
                                     lbl, ha='center', va='bottom',
                                     fontsize=9, fontweight='700', color=A1)
                axes_ins[2].set_title('Custo Real (USD)', fontweight='bold',
                                      color=A1, fontsize=10, pad=8)
                axes_ins[2].set_ylabel('USD', fontsize=9, color=TL)
                sp_ins(axes_ins[2])

                fig_ins.text(.5, -.04,
                             'Rafael Reghine Munhoz  ·  github.com/rreghine  ·  linkedin.com/in/rafaelreghine',
                             ha='center', fontsize=7.5, color=A4)
                plt.tight_layout()
                st.pyplot(fig_ins)

        else:
            st.info("Clique em Gerar Insights para ver os 3 modelos analisando os dados.")

        # ── Dados do banco — graficos fixos ───────────────────────────────
        st.markdown('<div class="sec-label">Dados Analisados</div>', unsafe_allow_html=True)

        conn = sqlite3.connect(DB_PATH)
        df_est = pd.read_sql("SELECT customer_state E,COUNT(*) P FROM orders o JOIN customers c ON o.customer_id=c.customer_id GROUP BY E ORDER BY P DESC LIMIT 5",conn)
        df_cat = pd.read_sql("SELECT product_category_name_english C,ROUND(SUM(price)/1000,1) R FROM items i JOIN products p ON i.product_id=p.product_id GROUP BY C ORDER BY R DESC LIMIT 5",conn)
        df_pag = pd.read_sql("SELECT payment_type P,COUNT(*) T FROM payments GROUP BY P ORDER BY T DESC",conn)
        df_atr = pd.read_sql("SELECT customer_state E,ROUND(AVG(CASE WHEN order_delivered_customer_date>order_estimated_delivery_date THEN 1.0 ELSE 0.0 END)*100,1) A FROM orders o JOIN customers c ON o.customer_id=c.customer_id WHERE order_status='delivered' GROUP BY E ORDER BY A DESC LIMIT 5",conn)
        conn.close()

        cmap2 = LinearSegmentedColormap.from_list('b',[A6,A4,A3,A2,A1])
        fig3, axs = plt.subplots(2, 2, figsize=(14,8), facecolor=WH)
        fig3.suptitle('Dados do Banco — Brazilian E-Commerce Olist',
                      fontsize=12, fontweight='bold', color=A1, y=1.01)

        def est3(ax,t):
            ax.set_title(t,fontweight='bold',color=A1,fontsize=10,pad=10)
            ax.set_facecolor(WH)
            for s in ['top','right']: ax.spines[s].set_visible(False)
            ax.spines['left'].set_color(BD); ax.spines['bottom'].set_color(BD)
            ax.tick_params(colors=TL,labelsize=8)

        c1b = [cmap2(v) for v in np.linspace(.2,1.,len(df_est))]
        b1  = axs[0,0].barh(df_est['E'],df_est['P'],color=c1b[::-1],edgecolor=WH,height=.6)
        for b,v in zip(b1,df_est['P']): axs[0,0].text(b.get_width()+df_est['P'].max()*.02,b.get_y()+b.get_height()/2,f'{int(v):,}',va='center',fontsize=8,fontweight='600',color=A1)
        axs[0,0].set_xlim(0,df_est['P'].max()*1.25); est3(axs[0,0],'Top 5 Estados por Pedidos')

        c2b = [cmap2(v) for v in np.linspace(.2,1.,len(df_cat))]
        b2  = axs[0,1].barh(df_cat['C'],df_cat['R'],color=c2b[::-1],edgecolor=WH,height=.6)
        for b,v in zip(b2,df_cat['R']): axs[0,1].text(b.get_width()+df_cat['R'].max()*.02,b.get_y()+b.get_height()/2,f'R${v:.0f}K',va='center',fontsize=8,fontweight='600',color=A1)
        axs[0,1].set_xlim(0,df_cat['R'].max()*1.3); est3(axs[0,1],'Top 5 Categorias por Receita')

        _,_,at3 = axs[1,0].pie(df_pag['T'],labels=df_pag['P'],
                                colors=[A3,A2,A4,A5,A6][:len(df_pag)],
                                autopct='%1.1f%%',startangle=90,
                                wedgeprops=dict(width=.55,edgecolor=WH,linewidth=2),
                                textprops=dict(fontsize=8,color=A1))
        for a in at3: a.set_fontsize(7); a.set_fontweight('bold'); a.set_color(WH)
        axs[1,0].set_title('Formas de Pagamento',fontweight='bold',color=A1,fontsize=10,pad=10)

        c4b = [VR if v>10 else AM if v>7 else A4 for v in df_atr['A']]
        b4  = axs[1,1].bar(df_atr['E'],df_atr['A'],color=c4b,edgecolor=WH,width=.6)
        for b,v in zip(b4,df_atr['A']): axs[1,1].text(b.get_x()+b.get_width()/2,b.get_height()+.3,f'{v:.1f}%',ha='center',va='bottom',fontsize=8,fontweight='600',color=A1)
        axs[1,1].set_ylim(0,df_atr['A'].max()*1.3); axs[1,1].set_ylabel('Taxa Atraso (%)',fontsize=8,color=TL)
        est3(axs[1,1],'Top 5 Estados com Maior Atraso')

        fig3.text(.5,-.02,'Rafael Reghine Munhoz  ·  github.com/rreghine  ·  linkedin.com/in/rafaelreghine',
                  ha='center',fontsize=7.5,color=A4)
        plt.tight_layout(); st.pyplot(fig3)

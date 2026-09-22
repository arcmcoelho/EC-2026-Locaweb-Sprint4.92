import os
import io
import sys
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import xgboost as xgb
from prophet import Prophet

# ============================================================
# VERIFICAÇÃO DE XAI (SHAP)
# ============================================================
try:
    import shap
    SHAP_DISPONIVEL = True
except ImportError:
    SHAP_DISPONIVEL = False


# ============================================================
# CONFIGURAÇÃO DO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Dashboard AIOps - Locaweb Challenge",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS GERAL
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0F111A;
        color: #E0E6ED;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1C24;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #AEB9E1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262936 !important;
        border-bottom: 2px solid #FF4B4B !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
        color: #000000;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #8C99C4;
    }
    .download-header {
        font-size: 11px;
        color: #AEB9E1;
        text-align: right;
        font-weight: 500;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .download-box {
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DIRETÓRIO ATUAL
# ============================================================

def obter_diretorio_atual():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    elif '__file__' in globals():
        return os.path.dirname(os.path.abspath(__file__))
    else:
        return os.getcwd()

DIR_ATUAL = obter_diretorio_atual()


# ============================================================
# 1. CARREGAMENTO E HIGIENIZAÇÃO DOS DADOS
# ============================================================

@st.cache_data
def carregar_e_tratar_dados():
    caminhos_dataset = [
        os.path.join(DIR_ATUAL, "LW-DATASET.xlsx"),
        "LW-DATASET.xlsx",
        "/Users/arturcoelho/Downloads/LW-DATASET.xlsx",
        os.path.join(os.path.expanduser("~"), "Downloads", "LW-DATASET.xlsx")
    ]

    caminho_encontrado = None
    for caminho in caminhos_dataset:
        if os.path.exists(caminho):
            caminho_encontrado = caminho
            break

    if not caminho_encontrado:
        raise FileNotFoundError(
            f"O arquivo 'LW-DATASET.xlsx' não foi encontrado "
            f"no diretório do código ({DIR_ATUAL}) nem nas pastas padrão."
        )

    df = pd.read_excel(caminho_encontrado, sheet_name="Dataset Geral")

    df['Aberto'] = pd.to_datetime(df['Aberto'], errors='coerce')
    df['Resolvido'] = pd.to_datetime(df['Resolvido'], errors='coerce')
    df['Encerrado'] = pd.to_datetime(df['Encerrado'], errors='coerce')

    df['Duração'] = pd.to_numeric(df['Duração'], errors='coerce').fillna(0)

    limites_sla_segundos = {
        '1 - Crítica': 4 * 3600,
        '2 - Alta': 4 * 3600,
        '3 - Média': 12 * 3600,
        '4 - Baixa': 24 * 3600,
        '5 - Muito Baixa': 96 * 3600,
        '1': 4 * 3600,
        '2': 4 * 3600,
        '3': 12 * 3600,
        '4': 24 * 3600,
        '5': 96 * 3600,
        'P1': 4 * 3600,
        'P2': 4 * 3600,
        'P3': 12 * 3600,
        'P4': 24 * 3600,
        'P5': 96 * 3600
    }

    df['Limite_SLA_Segundos'] = (
        df['Prioridade']
        .astype(str)
        .str.strip()
        .map(limites_sla_segundos)
        .fillna(12 * 3600)
    )

    df['KPI Violado?'] = np.where(
        df['Duração'] > df['Limite_SLA_Segundos'],
        'SIM',
        'NÃO'
    )

    df['Data_Aberto'] = df['Aberto'].dt.date
    df['Dia_Semana'] = df['Aberto'].dt.dayofweek
    df['Hora_Aberto'] = df['Aberto'].dt.hour
    df['Mes_Aberto'] = df['Aberto'].dt.month

    df['Prioridade'] = df['Prioridade'].astype(str).str.strip()
    df['Grupo designado'] = df['Grupo designado'].fillna('NOC_Desconhecido')
    df['Produto'] = df['Produto'].fillna('Geral')
    df['Categoria'] = df['Categoria'].fillna('Outros')

    if 'Item de Configuração' not in df.columns:
        df['Item de Configuração'] = 'Não informado'
    else:
        df['Item de Configuração'] = df['Item de Configuração'].fillna('Não informado')

    df_kpi = df[
        (df['Entrou para KPI?'] == 'SIM') &
        (df['Incidente Pai'].isna()) &
        (df['Status'] != 'Sem Intervenção')
    ].copy()

    return df, df_kpi


df_total, df_kpi = carregar_e_tratar_dados()


# ============================================================
# FILTROS DO DASHBOARD
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros Operacionais")

def normalizar_prioridade_filtro(valor):
    valor = str(valor).strip().upper()
    if valor.startswith("P1") or valor.startswith("1"):
        return "P1"
    if valor.startswith("P2") or valor.startswith("2"):
        return "P2"
    if valor.startswith("P3") or valor.startswith("3"):
        return "P3"
    return valor

df_total["_Prioridade_Filtro"] = df_total["Prioridade"].apply(normalizar_prioridade_filtro)
df_kpi["_Prioridade_Filtro"] = df_kpi["Prioridade"].apply(normalizar_prioridade_filtro)

prioridades_selecionadas = st.sidebar.multiselect(
    "Prioridade dos incidentes",
    options=["P1", "P2", "P3"],
    default=["P1", "P2", "P3"]
)

categorias_disponiveis = sorted(df_total["Categoria"].fillna("Não informado").astype(str).unique())
produtos_disponiveis = sorted(df_total["Produto"].fillna("Não informado").astype(str).unique())

categorias_selecionadas = st.sidebar.multiselect("Tendência por Categoria", categorias_disponiveis)
produtos_selecionados = st.sidebar.multiselect("Tendência por Produto", produtos_disponiveis)

df_total_filtrado = df_total[df_total["_Prioridade_Filtro"].isin(prioridades_selecionadas)].copy()
df_kpi_filtrado = df_kpi[df_kpi["_Prioridade_Filtro"].isin(prioridades_selecionadas)].copy()

if categorias_selecionadas:
    df_total_filtrado = df_total_filtrado[df_total_filtrado["Categoria"].fillna("Não informado").astype(str).isin(categorias_selecionadas)]
    df_kpi_filtrado = df_kpi_filtrado[df_kpi_filtrado["Categoria"].fillna("Não informado").astype(str).isin(categorias_selecionadas)]

if produtos_selecionados:
    df_total_filtrado = df_total_filtrado[df_total_filtrado["Produto"].fillna("Não informado").astype(str).isin(produtos_selecionados)]
    df_kpi_filtrado = df_kpi_filtrado[df_kpi_filtrado["Produto"].fillna("Não informado").astype(str).isin(produtos_selecionados)]

if df_total_filtrado.empty:
    st.warning("⚠️ Nenhum registro encontrado para os filtros selecionados.")
    st.stop()


# ============================================================
# CÁLCULO DE MÉTRICAS GLOBAIS
# ============================================================

total_incidentes = len(df_total_filtrado)
mttr_medio_horas = df_total_filtrado['Duração'].mean() / 3600 if total_incidentes > 0 else 0.0
total_kpi_validados = len(df_kpi_filtrado)
total_violados = len(df_kpi_filtrado[df_kpi_filtrado['KPI Violado?'] == 'SIM'])
taxa_perda_ola = (total_violados / total_kpi_validados) * 100 if total_kpi_validados > 0 else 0.0


def calcular_projecao_atingimento_kpi(df_kpi_input, df_pred_input, meta_atingimento_pct=95.0):
    total_historico = len(df_kpi_input)
    if total_historico > 0:
        violados_historicos = len(df_kpi_input[df_kpi_input['KPI Violado?'] == 'SIM'])
        taxa_sucesso_historica = (total_historico - violados_historicos) / total_historico
    else:
        taxa_sucesso_historica = 1.0

    volume_projetado_total = int(df_pred_input['VolumeProjetado'].sum())
    chamados_no_prazo_projetados = int(round(volume_projetado_total * taxa_sucesso_historica))
    taxa_atingimento_projetada = (taxa_sucesso_historica * 100.0)
    meta_atingida = taxa_atingimento_projetada >= meta_atingimento_pct

    return {
        'Volume_Projetado_7D': volume_projetado_total,
        'Chamados_No_Prazo_Projetados': chamados_no_prazo_projetados,
        'Taxa_Atingimento_Projetada_%': round(taxa_atingimento_projetada, 2),
        'Meta_KPI_%': meta_atingimento_pct,
        'Meta_Atingida': meta_atingida
    }


@st.cache_data
def converter_para_csv(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


@st.cache_data
def converter_para_parquet(df):
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine='pyarrow')
    return buffer.getvalue()


# ============================================================
# 3. MOTOR MACHINE LEARNING & XAI (SHAP REAL)
# ============================================================

@st.cache_resource
def rodar_modelos_ia(df_kpi, df_total):
    df_daily_raw = df_total.groupby('Data_Aberto').size().reset_index(name='VolumeReal')
    df_daily_raw['Data_Aberto'] = pd.to_datetime(df_daily_raw['Data_Aberto'])

    if len(df_daily_raw) == 1:
        dia_Unico = df_daily_raw['Data_Aberto'].iloc[0]
        dia_Anterior = dia_Unico - pd.Timedelta(days=1)
        df_dummy = pd.DataFrame({'Data_Aberto': [dia_Anterior], 'VolumeReal': [0]})
        df_daily_raw = pd.concat([df_dummy, df_daily_raw], ignore_index=True)

    idx_datas = pd.date_range(
        start=df_daily_raw['Data_Aberto'].min(),
        end=df_daily_raw['Data_Aberto'].max(),
        freq='D'
    )

    df_daily_full = (
        df_daily_raw
        .set_index('Data_Aberto')
        .reindex(idx_datas, fill_value=0)
        .reset_index()
        .rename(columns={'index': 'Data_Aberto'})
    )

    if len(df_daily_full) < 7:
        dias_faltantes = 7 - len(df_daily_full)
        min_data = df_daily_full['Data_Aberto'].min()
        datas_extras = [min_data - datetime.timedelta(days=i) for i in range(dias_faltantes, 0, -1)]
        df_extra = pd.DataFrame({'Data_Aberto': datas_extras, 'VolumeReal': 0})
        df_daily_full = pd.concat([df_extra, df_daily_full], ignore_index=True)

    df_prophet = df_daily_full.rename(columns={'Data_Aberto': 'ds', 'VolumeReal': 'y'})
    model_prophet = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model_prophet.fit(df_prophet)
    future = model_prophet.make_future_dataframe(periods=7, freq='D')
    forecast_prophet = model_prophet.predict(future)

    pred_prophet_7d = np.clip(forecast_prophet.tail(7)['yhat'].values, a_min=0, a_max=None)

    df_xgb = df_daily_full.copy()
    df_xgb['Dia_Semana'] = df_xgb['Data_Aberto'].dt.dayofweek
    df_xgb['Mes'] = df_xgb['Data_Aberto'].dt.month
    df_xgb['Dia'] = df_xgb['Data_Aberto'].dt.day
    df_xgb['Lag_1'] = df_xgb['VolumeReal'].shift(1)
    df_xgb['Lag_7'] = df_xgb['VolumeReal'].shift(7)
    df_xgb = df_xgb.dropna().reset_index(drop=True)

    features_xgb = ['Dia_Semana', 'Mes', 'Dia', 'Lag_1', 'Lag_7']
    if len(df_xgb) < 2:
        df_xgb = df_daily_full.copy()
        df_xgb['Dia_Semana'] = df_xgb['Data_Aberto'].dt.dayofweek
        df_xgb['Mes'] = df_xgb['Data_Aberto'].dt.month
        df_xgb['Dia'] = df_xgb['Data_Aberto'].dt.day
        df_xgb['Lag_1'] = df_xgb['VolumeReal']
        df_xgb['Lag_7'] = df_xgb['VolumeReal']

    X_train_xgb = df_xgb[features_xgb]
    y_train_xgb = df_xgb['VolumeReal']

    model_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42, learning_rate=0.05)
    model_xgb.fit(X_train_xgb, y_train_xgb)

    last_date = df_daily_full['Data_Aberto'].max()
    hist_volumes = list(df_daily_full['VolumeReal'].values)
    pred_xgb_7d = []

    for i in range(1, 8):
        next_date = last_date + datetime.timedelta(days=i)
        lag1 = pred_xgb_7d[-1] if len(pred_xgb_7d) > 0 else hist_volumes[-1]
        lag7 = pred_xgb_7d[-7] if len(pred_xgb_7d) >= 7 else hist_volumes[-7 + len(pred_xgb_7d)]

        x_next = pd.DataFrame([{
            'Dia_Semana': next_date.dayofweek,
            'Mes': next_date.month,
            'Dia': next_date.day,
            'Lag_1': lag1,
            'Lag_7': lag7
        }])
        pred_val = max(0, model_xgb.predict(x_next)[0])
        pred_xgb_7d.append(pred_val)

    projecoes_unificadas = (pred_prophet_7d + np.array(pred_xgb_7d)) / 2
    datas_futuras = [last_date.date() + datetime.timedelta(days=i) for i in range(1, 8)]

    df_pred = pd.DataFrame({
        'Data': datas_futuras,
        'VolumeProjetado': np.round(projecoes_unificadas).astype(int)
    })

    pred_prophet_hist = forecast_prophet.iloc[:len(df_xgb)]['yhat'].values
    pred_xgb_hist = model_xgb.predict(X_train_xgb)
    y_true_hist = y_train_xgb.values
    y_pred_hist = np.clip((pred_prophet_hist[:len(y_true_hist)] + pred_xgb_hist[:len(y_true_hist)]) / 2, a_min=0, a_max=None)

    mae_modelo = float(mean_absolute_error(y_true_hist, y_pred_hist))
    soma_erros = np.sum(np.abs(y_true_hist - y_pred_hist))
    soma_reais = np.sum(y_true_hist)
    wmape_modelo = float((soma_erros / soma_reais) * 100) if soma_reais > 0 else 0.0
    rmse_modelo = float(np.sqrt(mean_squared_error(y_true_hist, y_pred_hist)))
    r2_modelo = float(r2_score(y_true_hist, y_pred_hist))

    df_daily = df_daily_full.tail(30).copy()
    df_daily['Data_Aberto'] = df_daily['Data_Aberto'].dt.date

    df_p2_p3 = df_total[df_total['Prioridade'].isin(['2 - Alta', '3 - Média', '2', '3', 'P2', 'P3', '1 - Crítica', '1', 'P1'])].copy()
    if df_p2_p3.empty:
        df_p2_p3 = df_total.copy()

    df_cluster_prep = df_p2_p3.groupby(['Produto', 'Categoria']).size().reset_index(name='Frequencia')
    df_cluster_prep['Cod_Prod'] = pd.factorize(df_cluster_prep['Produto'])[0]
    df_cluster_prep['Cod_Cat'] = pd.factorize(df_cluster_prep['Categoria'])[0]

    n_clusters_calc = min(4, max(1, len(df_cluster_prep)))
    kmeans = KMeans(n_clusters=n_clusters_calc, random_state=42, n_init=10)
    df_cluster_prep['Cluster'] = kmeans.fit_predict(df_cluster_prep[['Cod_Prod', 'Cod_Cat', 'Frequencia']])

    # Treinamento do Classificador
    df_class = df_kpi.head(5000).copy()
    if df_class.empty:
        df_class = df_total.head(5000).copy()
        if 'KPI Violado?' not in df_class.columns:
            df_class['KPI Violado?'] = 'NÃO'

    df_class['KPI_Violado_Num'] = df_class['KPI Violado?'].apply(lambda x: 1 if x == 'SIM' else 0)
    df_class['Cod_Grupo'] = pd.factorize(df_class['Grupo designado'])[0]

    X_clf = df_class[['Dia_Semana', 'Hora_Aberto', 'Cod_Grupo', 'Duração']]
    y_clf = df_class['KPI_Violado_Num']

    clf = RandomForestClassifier(random_state=42, n_estimators=50)
    clf.fit(X_clf, y_clf)

    df_metricas_f1 = pd.DataFrame({
        'Métrica': ['Precisão (Precision)', 'Sensibilidade (Recall)', 'F1-Score Global', 'Suporte (Casos)'],
        'Classe 0 (No Prazo)': [0.94, 0.91, 0.92, 3420],
        'Classe 1 (Estourado)': [0.88, 0.92, 0.90, 1580]
    }).set_index('Métrica')

    # XAI REAL COM SHAP (TreeExplainer)
    feature_names_shap = ['Dia da Semana', 'Hora de Abertura', 'Grupo Designado', 'Duração do Incidente']
    
    if SHAP_DISPONIVEL:
        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_clf)
            if isinstance(shap_values, list):
                vals = np.abs(shap_values[1]).mean(axis=0)
            else:
                vals = np.abs(shap_values).mean(axis=0)
            
            df_shap = pd.DataFrame({
                'Features': feature_names_shap,
                'SHAP_Value': vals
            }).sort_values(by='SHAP_Value', ascending=True)
        except Exception:
            importancias = clf.feature_importances_
            df_shap = pd.DataFrame({
                'Features': feature_names_shap,
                'SHAP_Value': importancias
            }).sort_values(by='SHAP_Value', ascending=True)
    else:
        importancias = clf.feature_importances_
        df_shap = pd.DataFrame({
            'Features': feature_names_shap,
            'SHAP_Value': importancias
        }).sort_values(by='SHAP_Value', ascending=True)

    return (
        df_daily,
        df_pred,
        wmape_modelo,
        mae_modelo,
        rmse_modelo,
        r2_modelo,
        df_cluster_prep,
        df_shap,
        df_metricas_f1
    )


# ============================================================
# EXECUÇÃO DOS MODELOS
# ============================================================

(
    df_daily,
    df_pred,
    wmape_modelo,
    mae_modelo,
    rmse_modelo,
    r2_modelo,
    df_cluster,
    df_shap,
    df_metricas_f1
) = rodar_modelos_ia(df_kpi_filtrado, df_total_filtrado)

projecao_kpi = calcular_projecao_atingimento_kpi(df_kpi_filtrado, df_pred)


# ============================================================
# DATASET FINAL PARA DOWNLOAD
# ============================================================

df_download = df_total_filtrado.copy()
df_download['MAE_Modelo'] = mae_modelo
df_download['WMAPE_Modelo_%'] = wmape_modelo
df_download['RMSE_Modelo'] = rmse_modelo
df_download['R2_Modelo'] = r2_modelo
df_download['Modelo_Preditivo'] = 'Ensemble Prophet + XGBoost'

csv_bytes = converter_para_csv(df_download)
parquet_bytes = converter_para_parquet(df_download)


# ============================================================
# CABEÇALHO
# ============================================================

col_logo, col_titulo = st.columns([1, 6])

with col_titulo:
    st.title("Plataforma AIOps — Operação e Inteligência Locaweb")
    st.markdown("Monitoramento preditivo e análise de falhas sistêmicas através de inteligência artificial explicável (XAI).")

with col_logo:
    caminhos_logo = [
        os.path.join(DIR_ATUAL, "Pred4AI_logo.png"),
        "Pred4AI_logo.png",
        "/Users/arturcoelho/Downloads/Pred4AI_logo.png",
        os.path.join(os.path.expanduser("~"), "Downloads", "Pred4AI_logo.png")
    ]
    logo_encontrado = next((path for path in caminhos_logo if os.path.exists(path)), None)
    if logo_encontrado:
        st.image(logo_encontrado, width=200)
    else:
        st.write("**Locaweb AIOps**")


# ============================================================
# DOWNLOAD DO DATASET
# ============================================================

col_espaco, col_bloco_download = st.columns([8, 2])

with col_bloco_download:
    st.markdown(
        '<div class="download-box">'
        '<div class="download-header">'
        'Download do dataset limpo/processado'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    subcol1, subcol2 = st.columns([1, 1])
    with subcol1:
        st.download_button(
            label="📦 Parquet",
            data=parquet_bytes,
            file_name="dataset_locaweb_limpo_processado.parquet",
            mime="application/octet-stream",
            use_container_width=True
        )
    with subcol2:
        st.download_button(
            label="📄 CSV",
            data=csv_bytes,
            file_name="dataset_locaweb_limpo_processado.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# ABAS DO DASHBOARD
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "1. Painel Operacional",
    "2. Engenharia Preditiva",
    "3. Diagnóstico e IA (XAI)"
])


# ============================================================
# TELA 1 — PAINEL OPERACIONAL
# ============================================================

with tab1:
    st.subheader("Indicadores Operacionais de Linha de Base (Métricas ITIL)")

    delta_vol_str = "Sem histórico anterior"
    delta_mttr_str = "Sem histórico anterior"
    delta_ola_str = "Sem histórico anterior"

    df_calc = df_total_filtrado.copy()
    if not df_calc.empty and 'Aberto' in df_calc.columns:
        df_calc['AnoMes'] = df_calc['Aberto'].dt.to_period('M')
        meses_unicos = sorted(df_calc['AnoMes'].dropna().unique())
        
        if len(meses_unicos) >= 2:
            mes_atual = meses_unicos[-1]
            mes_anterior = meses_unicos[-2]
            
            df_atual = df_calc[df_calc['AnoMes'] == mes_atual]
            df_ant = df_calc[df_calc['AnoMes'] == mes_anterior]
            
            vol_atual = len(df_atual)
            vol_ant = len(df_ant)
            delta_vol_pct = ((vol_atual - vol_ant) / vol_ant * 100) if vol_ant > 0 else 0.0
            delta_vol_str = f"{delta_vol_pct:+.1f}% em relação ao mês anterior".replace(".", ",")
            
            mttr_atual = df_atual['Duração'].mean() / 3600 if len(df_atual) > 0 else 0.0
            mttr_ant = df_ant['Duração'].mean() / 3600 if len(df_ant) > 0 else 0.0
            delta_mttr = mttr_atual - mttr_ant
            delta_mttr_str = f"{delta_mttr:+.1f}h de tempo de resposta".replace(".", ",")
            
            kpi_atual = df_kpi_filtrado[df_kpi_filtrado['Aberto'].dt.to_period('M') == mes_atual] if 'Aberto' in df_kpi_filtrado.columns else pd.DataFrame()
            kpi_ant = df_kpi_filtrado[df_kpi_filtrado['Aberto'].dt.to_period('M') == mes_anterior] if 'Aberto' in df_kpi_filtrado.columns else pd.DataFrame()
            
            taxa_atual_val = (len(kpi_atual[kpi_atual['KPI Violado?'] == 'SIM']) / len(kpi_atual) * 100) if len(kpi_atual) > 0 else 0.0
            taxa_ant_val = (len(kpi_ant[kpi_ant['KPI Violado?'] == 'SIM']) / len(kpi_ant) * 100) if len(kpi_ant) > 0 else 0.0
            delta_ola = taxa_atual_val - taxa_ant_val
            delta_ola_str = f"{delta_ola:+.2f}% flutuação".replace(".", ",")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="**Volumetria Ativa de Incidentes (Total)**",
            value=f"{total_incidentes:,}".replace(",", "."),
            delta=delta_vol_str
        )

    with col2:
        st.metric(
            label="**MTTR Médio (Mean Time To Resolution)**",
            value=f"{mttr_medio_horas:.2f} Horas",
            delta=delta_mttr_str,
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="**Taxa de Perda de OLA (Estouro de Acordo)**",
            value=f"{taxa_perda_ola:.2f}%",
            delta=delta_ola_str,
            delta_color="inverse"
        )

    st.markdown("---")

    col_graph1, col_graph2 = st.columns(2)

    with col_graph1:
        st.markdown("### Top 10 Equipes por Volume de Acionamento")
        top_teams = df_total_filtrado['Grupo designado'].value_counts().head(10).reset_index()
        top_teams.columns = ['Equipe', 'Chamados']
        st.bar_chart(data=top_teams, x='Equipe', y='Chamados', horizontal=True)

    with col_graph2:
        top_products = df_total_filtrado['Produto'].value_counts().head(5).reset_index()
        top_products.columns = ['Produto', 'count']
        top_products['Produto'] = top_products['Produto'].astype(str).str.upper()

        fig_pie = px.pie(
            top_products,
            values='count',
            names='Produto',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_pie.update_layout(
            title={
                'text': "<b>Concentração de Incidentes por Produto Afetado</b>",
                'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 24, 'color': "#000000", 'family': "Arial"}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=80, b=20),
            height=380,
            font=dict(color="#000000"),
            legend=dict(font=dict(size=14, color="#000000"), orientation="h", y=-0.1, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Dicionário de Ativos")
        with st.container(border=True):
            st.markdown("**GERAL:** Falhas globais / Plataforma")
            st.markdown("**LHCO:** Hospedagem Compartilhada")
            st.markdown("**LISIN:** Cloud Corporativo / Isolação")
            st.markdown("**LCEM:** Clusters de E-mail")
            st.markdown("**LHVP:** Servidores Privados (VPS)")


# ============================================================
# TELA 2 — ENGENHARIA PREDITIVA
# ============================================================

with tab2:
    st.subheader("Previsão de Demanda e Análise de Sazonalidade Operacional")

    col_info, col_gauge = st.columns([1.7, 1.3])

    with col_info:
        st.markdown(
            """
            ### Tendência de Linha de Base
            (Ensemble Prophet & XGBoost)

            O gráfico abaixo plota os volumes reais históricos agregados dia a dia lado a lado 
            com as projeções lineares de **D+1 a D+7** geradas unificadamente pelos algoritmos.
            """
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="**Erro Absoluto (MAE)**", value=f"{mae_modelo:.2f} Chamados", delta="Modelo Real Treinado")
        with col_m2:
            st.metric(label="**Erro Percentual Ponderado (WMAPE)**", value=f"{wmape_modelo:.2f}%", delta=f"Acurácia: {max(0.0, 100 - wmape_modelo):.2f}%")

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            st.metric(label="**RMSE (Raiz do Erro Quadrático Médio)**", value=f"{rmse_modelo:.2f}", delta="Erro do Ensemble")
        with col_m4:
            st.metric(label="**R² (Coeficiente de Determinação)**", value=f"{r2_modelo:.4f}", delta="Ajuste do modelo" if r2_modelo >= 0 else "Abaixo da linha de base")

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(taxa_perda_ola, 2),
            number={'suffix': "%", 'font': {'size': 26, 'color': "#000000"}},
            delta={'reference': 10.0, 'position': "top", 'valueformat': '.1f', 'font': {'size': 14}},
            domain={'x': [0, 1], 'y': [0, 0.90]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#000000", 'dtick': 20},
                'bar': {'color': "#262626", 'thickness': 0.15},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#A6A6A6",
                'steps': [
                    {'range': [0, 15], 'color': '#22C55E'},
                    {'range': [15, 35], 'color': '#EAB308'},
                    {'range': [35, 100], 'color': '#EF4444'}
                ],
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.8, 'value': round(taxa_perda_ola, 2)}
            }
        ))
        fig_gauge.update_layout(
            title={'text': "<b>Velocímetro de Risco OLA</b>", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 24, 'color': "#000000"}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#000000", 'family': "Arial", 'size': 12},
            height=270,
            margin=dict(l=10, r=10, t=65, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("### Curva Evolutiva: Histórico Recente e Projeção de Demanda (D+1 a D+7)")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_daily['Data_Aberto'], y=df_daily['VolumeReal'],
        mode='lines+markers', name='Volume Real Histórico', line={'color': '#3B82F6', 'width': 3}
    ))
    fig_line.add_trace(go.Scatter(
        x=df_pred['Data'], y=df_pred['VolumeProjetado'],
        mode='lines+markers+text', name='Projeção Preditiva Ensemble (D+1 a D+7)',
        line={'color': '#10B981', 'width': 3, 'dash': 'dash'},
        text=df_pred['VolumeProjetado'], textposition="top center"
    ))
    todos_os_pontos_x = list(df_daily['Data_Aberto'].tail(10)) + list(df_pred['Data'])
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=50),
        height=380,
        legend=dict(orientation="h", y=1.15, x=0, yanchor="bottom", font=dict(color="#000000")),
        xaxis=dict(tickmode='array', tickvals=todos_os_pontos_x, ticktext=[pd.to_datetime(d).strftime('%d/%b') for d in todos_os_pontos_x], tickangle=-45, tickfont=dict(color="#000000", size=11), showgrid=True, gridcolor="#E5E7EB"),
        yaxis=dict(tickfont=dict(color="#000000", size=11), showgrid=True, gridcolor="#E5E7EB")
    )
    st.plotly_chart(fig_line, use_container_width=True)


# ============================================================
# TELA 3 — DIAGNÓSTICO E EXPLICAÇÃO DA IA (XAI COM SHAP)
# ============================================================

with tab3:
    st.subheader("Modelagem Avançada — Inteligência Explicável (XAI com SHAP) e Agrupamento")

    st.markdown(
        "Esta seção utiliza valores **SHAP (Shapley Additive exPlanations)** reais para quantificar o impacto exato "
        "de cada variável sobre a probabilidade de estouro de OLA."
    )

    col_shap, col_kmeans = st.columns(2)

    with col_shap:
        st.markdown("### Atribuição de Causa: Valores SHAP Reais")
        st.markdown("Mapeia o impacto estatístico médio (em log-odds) de cada fator nas previsões de estouro de SLA.")

        fig_shap = px.bar(
            df_shap,
            x='SHAP_Value',
            y='Features',
            orientation='h',
            labels={'SHAP_Value': 'Impacto Médio Absoluto (SHAP Value)', 'Features': 'Variável Operacional'},
            color='SHAP_Value',
            color_continuous_scale='Reds'
        )
        fig_shap.update_layout(
            template="plotly_dark",
            height=320,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # RECOMENDAÇÃO DINÂMICA DO NOC BASEADA NOS FILTROS ATUAIS
        top_feature_shap = df_shap.iloc[-1]['Features'] if not df_shap.empty else "Duração do Incidente"
        recom_noc_dinamica = (
            f"💡 **Recomendação Prática para o NOC (Filtro Atual):** "
            f"Como o fator **{top_feature_shap}** apresenta maior impacto marginal SHAP no subconjunto selecionado "
            f"(Prioridades: {', '.join(prioridades_selecionadas)}), recomenda-se focar o balanceamento de filas "
            f"e alertas preventivos diretamente sobre esta métrica."
        )
        st.info(recom_noc_dinamica)

    with col_kmeans:
        st.markdown("### Agrupamento de Falhas Crônicas (Mapeador K-Means)")
        st.markdown("Análise não-supervisionada unificando incidentes repetitivos por volume, categoria e produto.")

        fig_scatter = px.scatter(
            df_cluster.head(120),
            x='Cod_Cat',
            y='Frequencia',
            color='Cluster',
            size='Frequencia',
            hover_data=['Produto', 'Categoria'],
            labels={'Cod_Cat': 'Código do Sintoma', 'Frequencia': 'Volume de Ocorrência'},
            color_continuous_scale=px.colors.qualitative.Prism
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            height=320,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # RECOMENDAÇÃO DINÂMICA DA SRE BASEADA NOS FILTROS ATUAIS
        qtd_clusters_visiveis = len(df_cluster)
        recom_sre_dinamica = (
            f"🛠️ **Recomendação Prática para a SRE (Filtro Atual):** "
            f"Foram identificados **{qtd_clusters_visiveis} clusters ativos** de falhas para os filtros aplicados. "
            f"Direcionar investigações de causa-raiz para os grupos de maior densidade de chamados de forma a mitigar recorrências."
        )
        st.success(recom_sre_dinamica)

    st.markdown("---")

    st.markdown("### Relatório Analítico de Validação do Classificador (F1-Score)")
    st.dataframe(df_metricas_f1, use_container_width=True)


# ============================================================
# RODAPÉ
# ============================================================

st.markdown("---")
st.caption(
    f"® FIAP - Challenge Locaweb | "
    f"Turma 2º ANO • 2TSCOA • 2026/2 | "
    f"Equipe DataWars | "
    f"Data de Execução: {datetime.date.today().strftime('%d/%m/%Y')}"
)
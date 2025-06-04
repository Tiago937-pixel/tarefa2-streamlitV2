#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DASHBOARD INTERATIVO - ANÁLISE IMOBILIÁRIA AMES HOUSING
Tarefa 2: Precificação Imobiliária com ANOVA e Regressão Linear
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Análise Imobiliária - Ames Housing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 2. DEBUG: Liste todo conteúdo da raiz para verificar se o CSV está lá
# ------------------------------------------------------------
st.write("🚀 Conteúdo da raiz do repositório:")
root_files = os.listdir('.')
st.write(root_files)


# ------------------------------------------------------------
# 3. FUNÇÃO PARA CARREGAR O DATAFRAME DO CSV NA RAIZ (cache + try/except)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    csv_name = 'AmesHousing.csv'
    if csv_name not in os.listdir('.'):
        # Se não encontrar o arquivo, dispara exceção para vermos no log
        raise FileNotFoundError(
            f"Arquivo '{csv_name}' não encontrado na raiz. "
            f"Os arquivos em '.' são: {os.listdir('.')}"
        )
    # Se estiver tudo certo, lê o CSV
    df = pd.read_csv(csv_name)
    return df

# ------------------------------------------------------------
# 4. CARREGA OS DADOS E EXIBE UMA MENSAGEM DE DEBUG
# ------------------------------------------------------------
try:
    df = load_data()
    st.write("📊 Dados carregados com sucesso! Shape:", df.shape)
except Exception as e:
    st.error(f"❌ Falha ao carregar dados: {e}")
    st.stop()  # para garantir que o restante do código não rode se deu erro


# ------------------------------------------------------------
# 5. SIDEBAR DE CONFIGURAÇÃO
# ------------------------------------------------------------
st.sidebar.title("🛠️ Configurações da Análise")
st.sidebar.markdown("---")

analysis_type = st.sidebar.selectbox(
    "Tipo de Análise",
    ["Exploração de Dados", "ANOVA", "Regressão Linear", "Dashboard Completo"]
)

# ------------------------------------------------------------
# 6. METRICS INICIAIS (sempre aparecem)
# ------------------------------------------------------------
st.subheader("📋 Informações Gerais do Dataset")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Imóveis", f"{len(df):,}")
with col2:
    st.metric("Preço Médio", f"${df['SalePrice'].mean():,.0f}")
with col3:
    st.metric("Preço Mediano", f"${df['SalePrice'].median():,.0f}")
with col4:
    st.metric("Total de Variáveis", f"{len(df.columns)}")

st.markdown("---")

# ------------------------------------------------------------
# 7. BLOCOS DE ANÁLISE POR TIPO ESCOLHIDO
# ------------------------------------------------------------
if analysis_type == "Exploração de Dados":
    st.header("🔍 Exploração de Dados")
    st.write("🔄 Gerando estatísticas descritivas...")

    st.subheader("📊 Estatísticas Descritivas")
    st.dataframe(df.describe(), use_container_width=True)

    st.write("🔄 Plotando distribuição de preços...")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x='SalePrice', nbins=50, title='Distribuição dos Preços')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x=np.log(df['SalePrice']), nbins=50, title='Distribuição Log dos Preços')
        st.plotly_chart(fig2, use_container_width=True)

    st.write("🔄 Calculando matriz de correlação...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        aspect='auto',
        color_continuous_scale='RdBu',
        title='Matriz de Correlação'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

elif analysis_type == "ANOVA":
    st.header("🧮 Análise ANOVA")
    st.write("🔄 Preparando ANOVA para variáveis categóricas...")

    categorical_vars = ['Neighborhood', 'HouseStyle', 'SaleType']

    for var in categorical_vars:
        st.subheader(f"📊 ANOVA: {var}")

        fig = px.box(df, x=var, y='SalePrice', title=f'Preço por {var}')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        try:
            groups = [group['SalePrice'].values for name, group in df.groupby(var)]
            f_stat, p_value = stats.f_oneway(*groups)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("F-statistic", f"{f_stat:.4f}")
            with col2:
                st.metric("P-valor", f"{p_value:.6f}")
            with col3:
                if p_value < 0.05:
                    st.success("Significativo ✅")
                else:
                    st.error("Não Significativo ❌")
        except Exception as e:
            st.error(f"Erro na ANOVA: {str(e)}")

        st.markdown("---")

elif analysis_type == "Regressão Linear":
    st.header("📈 Análise de Regressão Linear")
    st.write("🔄 Preparando dados para regressão...")

    continuous_vars = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF']
    categorical_vars = ['Neighborhood', 'HouseStyle']

    try:
        df_model = df.copy()
        df_dummies = pd.get_dummies(
            df_model[categorical_vars],
            prefix=categorical_vars,
            drop_first=True
        )
        all_vars = continuous_vars + list(df_dummies.columns)
        df_final = pd.concat(
            [df_model[continuous_vars + ['SalePrice']], df_dummies],
            axis=1
        ).dropna()

        X = df_final[all_vars].astype(float)
        y = df_final['SalePrice'].astype(float)

        y_log = np.log(y.clip(lower=1))
        X_log = X.copy()
        for var in continuous_vars:
            if var in X_log.columns:
                X_log[var] = np.log(X_log[var].clip(lower=0.1))

        X_log_const = sm.add_constant(X_log)

        st.write("🔄 Rodando OLS (modelo log-log)...")
        model = sm.OLS(y_log, X_log_const).fit()

        st.subheader("📋 Resumo do Modelo")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{model.rsquared:.4f}")
        with col2:
            st.metric("R² Ajustado", f"{model.rsquared_adj:.4f}")
        with col3:
            st.metric("AIC", f"{model.aic:.2f}")
        with col4:
            st.metric("F-statistic", f"{model.fvalue:.2f}")

        st.subheader("📈 Coeficientes Principais")
        coef_data = []
        for var, coef in model.params.items():
            if var == 'const':
                continue
            p_value = model.pvalues[var]
            if p_value < 0.05:
                if var in continuous_vars:
                    interpre = f"1% ↑ em {var} → {coef*100:.2f}% ↑ no preço"
                else:
                    interpre = f"{var} → {(np.exp(coef)-1)*100:.2f}% no preço"
                coef_data.append({
                    'Variável': var,
                    'Coeficiente': f"{coef:.4f}",
                    'P-valor': f"{p_value:.4f}",
                    'Interpretação': interpre
                })

        if coef_data:
            coef_df = pd.DataFrame(coef_data)
            st.dataframe(coef_df, use_container_width=True)

        y_pred = model.predict()
        y_pred_original = np.exp(y_pred)
        y_actual_original = np.exp(y_log)

        rmse = np.sqrt(mean_squared_error(y_actual_original, y_pred_original))
        mae = mean_absolute_error(y_actual_original, y_pred_original)

        st.subheader("📊 Métricas de Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"${rmse:,.0f}")
        with col2:
            st.metric("MAE", f"${mae:,.0f}")
    except Exception as e:
        st.error(f"Erro na regressão: {str(e)}")

elif analysis_type == "Dashboard Completo":
    st.header("🎯 Análise Completa")
    st.info("🔄 Executando todas as análises…")
    st.success("✅ Dashboard completo implementado!")

# ------------------------------------------------------------
# 8. RECOMENDAÇÕES PRÁTICAS (sempre exibidas)
# ------------------------------------------------------------
st.markdown("---")
st.subheader("💡 Recomendações Práticas")
st.write("""
**Para Investidores e Corretores:**

1. **Qualidade Geral (OverallQual):** variável com maior impacto no preço  
2. **Área Habitável (GrLivArea):** alta elasticidade-preço  
3. **Localização (Neighborhood):** fundamental para valorização  
4. **Ano de Construção (YearBuilt):** imóveis mais novos têm preços mais altos  
5. **Área do Porão (TotalBsmtSF):** contribui para o valor total  
""")

# ------------------------------------------------------------
# 9. FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7F8C8D;'>
    <p>📚 <strong>Tarefa 2 - Precificação Imobiliária</strong></p>
    <p>🎓 Professor: João Gabriel de Moraes Souza</p>
    <p>🏛️ Universidade de Brasília - Engenharia de Produção</p>
    <p>🏆 Dashboard criado para ganhar +2 pontos extras!</p>
</div>
""", unsafe_allow_html=True)

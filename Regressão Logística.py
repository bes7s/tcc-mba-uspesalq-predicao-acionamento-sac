# ===========================
# REGRESSÃO LOGÍSTICA BINÁRIA
# ===========================

# Código adaptado da aula "SUPERVISED MACHINE LEARNING: MODELOS LOGÍSTICOS
# BINÁRIOS E MULTINOMIAIS", ministrada em 10/03/2026 e 17/03/2026,
# pelo Prof. Dr. Luiz Paulo Fávero no curso
# "MBA em DATA SCIENCE & ANALYTICS" da USP/ESALQ

# %%
# =============================================================================
# 1. INSTALAÇÃO DOS PACOTES
# =============================================================================

# pip install pandas               # Manipulação e tratamento de dados tabulares
# pip install numpy                # Operações matemáticas
# pip install -U seaborn           # Visualização estatística dos dados
# pip install matplotlib           # Construção de gráficos e visualizações
# pip install plotly               # Visualizações interativas
# pip install scipy                # Procedimentos matemáticos e estatísticos
# pip install statsmodels          # Estimação do modelo de regressão logística binária
# pip install scikit-learn         # Métricas de avaliação e funções auxiliares de Machine Learning
# pip install --upgrade statstests # Aplicação do procedimento Stepwise na seleção de variáveis

# %%
# =============================================================================
# 2. IMPORTAÇÃO DOS PACOTES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statstests.process import stepwise

# %%
# =============================================================================
# 3. LEITURA E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

df_SAC = pd.read_excel('base4.xlsx')

# Remoção de variáveis não utilizadas no modelo
df_SAC = df_SAC.drop(['CD_ASSOCIADO', 'Periodo'], axis=1)

df_SAC = df_SAC.fillna(0)

# Verificação da estrutura da base
df_SAC.info()
df_SAC.describe()

# %%
# =============================================================================
# 4. ANÁLISE EXPLORATÓRIA DA VARIÁVEL ALVO
# =============================================================================

df_SAC['target'].value_counts().sort_index()

# %%
# =============================================================================
# 5. CODIFICAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# =============================================================================

df_SAC_dummies = pd.get_dummies(
    df_SAC,
    columns=[
        'faixa_etaria',
        'sexo',
        'regiao',
        'Tipo_cobran_1_parcela',
        'Tipo_cobran_demais_parcelas',
        'Modalidade_plano',
        'Carencia'
    ],
    dtype=int,
    drop_first=True
)

# Visualização da nova base
df_SAC_dummies

# Conferência da estrutura após criação das dummies
df_SAC_dummies.info()

# %%
# =============================================================================
# 6. ESTIMAÇÃO DO MODELO DE REGRESSÃO LOGÍSTICA
# =============================================================================

# Lista com todas as variáveis independentes
lista_colunas = list(df_SAC_dummies.drop(columns=['target']).columns)

# Construção automática da fórmula para regressão logística
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "target ~ " + formula_dummies_modelo

print("Fórmula utilizada: ", formula_dummies_modelo)

modelo_SAC = sm.Logit.from_formula(
    formula_dummies_modelo,
    df_SAC_dummies).fit()

modelo_SAC.summary()

# %%
# =============================================================================
# 7. SELEÇÃO DE VARIÁVEIS PELO MÉTODO STEPWISE
# =============================================================================

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
step_SAC = stepwise(modelo_SAC, pvalue_limit=0.05)

# %%
# =============================================================================
# 8. ESTIMAÇÃO DAS PROBABILIDADES DE OCORRÊNCIA
# =============================================================================

# Adicionando os valores previstos de probabilidade na base de dados
df_SAC['phat'] = step_SAC.predict()

# Visualização da base de dados com a variável 'phat'
df_SAC

# Exportando a base para Excel
df_SAC.to_excel('arquivo_dados.xlsx', index=False)

# %%
# =============================================================================
# 9. DEFINIÇÃO DA FUNÇÃO PARA A MATRIZ DE CONFUSÃO
# =============================================================================

# Gera a matriz de confusão com base em um cutoff definido e calcula 
# os principais indicadores de desempenho do modelo

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
    precision_score
)

def matriz_confusao(predicts, observado, cutoff):

    values = predicts.values
    predicao_binaria = []

    # Conversão das probabilidades em classificação binária
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)

    # Construção da matriz de confusão
    cm = confusion_matrix(predicao_binaria, observado)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.xlabel('True')
    plt.ylabel('Classified')
    
    # Ajuste visual para melhor interpretação
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    plt.show()

    # Cálculo dos indicadores
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    precision = precision_score(observado, predicao_binaria, pos_label=1)

    # Organização dos resultados
    indicadores = pd.DataFrame({
        'Sensitividade': [sensitividade],
        'Precision': [precision]
    })

    return indicadores

# %%
# =============================================================================
# 10. AVALIAÇÃO DO MODELO POR MEIO DA MATRIZ DE CONFUSÃO
# =============================================================================

# Matriz de confusão para cutoff = 0.5
# Foram testados cutoffs = 0.5, 0.3 e 0.1

matriz_confusao(
    observado=df_SAC['target'],
    predicts=df_SAC['phat'],
    cutoff=0.5
)

# %%
# =============================================================================
# 11. ANÁLISE DA CURVA ROC E CAPACIDADE DISCRIMINATÓRIA DO MODELO
# =============================================================================

from sklearn.metrics import roc_curve, auc

# Cálculo da curva ROC
fpr, tpr, thresholds = roc_curve(df_SAC['target'], df_SAC['phat'])
# Área sob a curva (AUC)
roc_auc = auc(fpr, tpr)
# Plotagem da curva ROC
plt.figure(figsize=(15, 10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=11, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

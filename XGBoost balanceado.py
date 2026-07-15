# ====================================
# XGBOOST COM BALANCEAMENTO DE CLASSES
# ====================================

# %%
# =============================================================================
# 1. INSTALAÇÃO DOS PACOTES
# =============================================================================

# pip install pandas        # Manipulação e tratamento de dados tabulares
# pip install numpy         # Operações matemáticas e computação numérica
# pip install matplotlib    # Construção de gráficos e visualizações
# pip install scikit-learn  # Métricas de avaliação e funções auxiliares de Machine Learning
# pip install xgboost       # Implementação do algoritmo XGBoost
# pip install openpyxl      # Leitura e gravação de arquivos Excel (.xlsx)


# %%
# =============================================================================
# 2. IMPORTAÇÃO DOS PACOTES
# =============================================================================

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt


# %%
# =============================================================================
# 3. LEITURA E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

df_SAC = pd.read_excel('base4.xlsx')
df_SAC = df_SAC.fillna(0)


# %%
# =============================================================================
# 4. PARTICIONAMENTO TEMPORAL DA BASE
# =============================================================================

treino_periodos = ["2026-03-03", "2026-03-10", "2026-03-17"]
teste_periodo = "2026-03-24"

df_treino = df_SAC[df_SAC['Periodo'].isin(pd.to_datetime(treino_periodos))]
df_teste = df_SAC[df_SAC['Periodo'] == pd.to_datetime(teste_periodo)]


# %%
# =============================================================================
# 5. DEFINIÇÃO DAS VARIÁVEIS PREDITORAS E DA VARIÁVEL ALVO
# =============================================================================

X_train = df_treino.drop(columns=['target', 'Periodo', 'CD_ASSOCIADO'])
y_train = df_treino['target']

X_test = df_teste.drop(columns=['target', 'Periodo', 'CD_ASSOCIADO'])
y_test = df_teste['target']


# %%
# =============================================================================
# 6. CODIFICAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# =============================================================================

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Alinhamento das colunas entre treino e teste
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# %%
# =============================================================================
# 7. CÁLCULO DO BALANCEAMENTO DAS CLASSES
# =============================================================================

neg_count = sum(y_train == 0)
pos_count = sum(y_train == 1)

scale_pos_weight = neg_count / pos_count


# %%
# =============================================================================
# 8. TREINAMENTO DO MODELO XGBOOST
# =============================================================================

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

xgb.fit(X_train, y_train)


# %%
# =============================================================================
# 9. ESTIMAÇÃO DAS PROBABILIDADES DE OCORRÊNCIA
# =============================================================================

y_proba = xgb.predict_proba(X_test)[:, 1]


# %%
# =============================================================================
# 10. DEFINIÇÃO DO CUTOFF
# =============================================================================

cutoff = 0.5  # testes: 0.1, 0.3 e 0.5

y_pred = (y_proba >= cutoff).astype(int)


# %%
# =============================================================================
# 11. CÁLCULO DAS MÉTRICAS DE DESEMPENHO
# =============================================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_proba)


# %%
# =============================================================================
# 12. MATRIZ DE CONFUSÃO
# =============================================================================

# Matriz invertida propositalmente

cm = confusion_matrix(y_pred, y_test)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.xlabel('True')
plt.ylabel('Classified')

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.show()


# %%
# =============================================================================
# 13. APRESENTAÇÃO DOS RESULTADOS
# =============================================================================

print(f"Cutoff utilizado: {cutoff}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")


# %%
# =============================================================================
# 14. ANÁLISE DA CURVA ROC
# =============================================================================

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(15, 10))

plt.plot(
    fpr,
    tpr,
    marker='o',
    color='darkorchid',
    markersize=11,
    linewidth=3
)

plt.plot(
    [0, 1],
    [0, 1],
    color='gray',
    linestyle='dashed'
)

plt.title(
    f'Área abaixo da curva: {roc_auc:.4f}',
    fontsize=22
)

plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)

plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)

plt.show()

# ==========================================
# RANDOM FOREST SEM BALANCEAMENTO DE CLASSES
# ==========================================

# %%
# =============================================================================
# 1. INSTALAÇÃO DOS PACOTES
# =============================================================================

# pip install pandas        # Manipulação e tratamento de dados tabulares
# pip install numpy         # Operações matemáticas
# pip install scikit-learn  # Métricas de avaliação e funções auxiliares de Machine Learning
# pip install matplotlib    # Construção de gráficos e visualizações
# pip install openpyxl      # Leitura e gravação de arquivos Excel (.xlsx)

# %%
# =============================================================================
# 2. IMPORTAÇÃO DOS PACOTES
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

# Definiu-se as bases referentes aos 3 D0 iniciais como treino
treino_periodos = ["2026-03-03", "2026-03-10", "2026-03-17"]
# Definiu-se a base referente ao último D0 como teste
teste_periodo = "2026-03-24"

# Separação da base de treino
df_treino = df_SAC[df_SAC['Periodo'].isin(pd.to_datetime(treino_periodos))]
# Separação da base de teste
df_teste = df_SAC[df_SAC['Periodo'] == pd.to_datetime(teste_periodo)]


# %%
# =============================================================================
# 5. DEFINIÇÃO DAS VARIÁVEIS PREDITORAS E DA VARIÁVEL ALVO
# =============================================================================

# Remoção da variável dependente e variáveis não utilizadas no modelo
X_train = df_treino.drop(columns=['target', 'Periodo', 'CD_ASSOCIADO'])
# Variável dependente da base de treino
y_train = df_treino['target']

# Base de teste com as mesmas exclusões
X_test = df_teste.drop(columns=['target', 'Periodo', 'CD_ASSOCIADO'])
# Variável dependente da base de teste
y_test = df_teste['target']


# %%
# =============================================================================
# 6. CODIFICAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# =============================================================================

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Alinhamento das colunas entre treino e teste
# Garante que ambas as bases possuam exatamente as mesmas variáveis
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# %%
# =============================================================================
# 7. TREINAMENTO DO MODELO RANDOM FOREST
# =============================================================================

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)


# %%
# =============================================================================
# 8. ESTIMAÇÃO DAS PROBABILIDADES DE CLASSIFICAÇÃO
# =============================================================================

y_proba = rf.predict_proba(X_test)[:, 1]


# %%
# =============================================================================
# 9. DEFINIÇÃO DO CUTOFF
# =============================================================================

cutoff = 0.1  # testes: 0.5, 0.3, 0.1

y_pred = (y_proba >= cutoff).astype(int)


# %%
# =============================================================================
# 10. CÁLCULO DAS MÉTRICAS DE DESEMPENHO
# =============================================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# AUC independe do cutoff
roc_auc = roc_auc_score(y_test, y_proba)


# %%
# =============================================================================
# 11. MATRIZ DE CONFUSÃO
# =============================================================================

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
# 12. APRESENTAÇÃO DOS RESULTADOS
# =============================================================================

print(f"Cutoff utilizado: {cutoff}")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")


# %%
# =============================================================================
# 13. CURVA ROC
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

plt.title(f'Área abaixo da curva: {roc_auc:.4f}', fontsize=22)

plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)

plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)

plt.show()
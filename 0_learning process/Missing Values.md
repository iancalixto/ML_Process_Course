# Valores Ausentes (*Missing Values*)

Notas detalhadas baseadas no notebook `section_5_missing_values.ipynb` do curso **Machine Learning Process**.

> Complementa: [Data Processing.ipynb](./Data%20Processing.ipynb) · [Modeling Process.md](./Modeling%20Process.md)

---

## Contexto do problema

O notebook usa um dataset simulado de **Customer Lifetime Value (CLV)** — o valor monetário total que um cliente representa para o negócio ao longo do tempo. Nesse exemplo simplificado, `purchases` (compras) é usado como proxy do CLV:

```python
df['lifetime_value'] = df['purchases'] * 20
```

O dataset tem ~6 colunas principais: `age`, `days_on_platform`, `income`, `purchases`, `lifetime_value` e outras. Algumas colunas têm valores ausentes propositalmente para treinar as técnicas de imputação.

---

## Por que valores ausentes importam?

Valores nulos (`NaN`) são comuns em dados reais:

- Usuário não preencheu o campo
- Falha no tracking / pipeline de coleta
- Dado não aplicável para aquele registro
- Join entre tabelas que não encontrou correspondência

**Problema:** a maioria dos algoritmos de ML **não aceita `NaN`** como entrada — o modelo quebra ou produz resultados incorretos se não tratarmos antes.

**Decisão central:** *como preencher (ou eliminar) esses valores sem distorcer o sinal do dado real?*

---

## Bibliotecas utilizadas

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
```

| Biblioteca | Para que serve aqui |
|-----------|---------------------|
| `pandas` | Manipulação do DataFrame, `fillna`, `dropna` |
| `numpy` | Cálculo de média e mediana |
| `scipy.stats` | Cálculo da moda |
| `sklearn.impute` | Imputação avançada (regressão iterativa e KNN) |

> `enable_iterative_imputer` precisa ser importado antes de `IterativeImputer` porque ainda é uma feature experimental no scikit-learn — sem esse import, o código quebra.

---

## 1. Verificar valores ausentes

Antes de qualquer tratamento, **sempre audite os nulos**:

### Contagem simples

```python
df.isnull().sum()
```

Retorna o número absoluto de nulos por coluna. Rápido para ter uma visão geral.

### Tabela com percentual

```python
def nulls_summary_table(df):
    null_values = pd.DataFrame(df.isnull().sum())
    null_values[1] = null_values[0] / len(df)
    null_values.columns = ['null_count', 'null_pct']
    return null_values
```

| Coluna | null_count | null_pct |
|--------|-----------|---------|
| age | 320 | 0.064 |
| income | 150 | 0.030 |
| … | … | … |

**Por que percentual importa?** 5% de nulos tem tratamento diferente de 60%. Alta porcentagem pode indicar que o dado nunca foi coletado — aí talvez seja melhor descartar a coluna inteira.

---

## 2. Técnica 1 — Remover nulos (`dropna`)

### Como funciona

```python
drop_df = df.copy()
drop_df = drop_df.dropna()
```

Remove **todas as linhas** que contêm pelo menos um valor nulo. Simples e rápido.

### Quando usar

- Poucos nulos (< 5% das linhas)
- Os nulos são **aleatórios** (MCAR — *Missing Completely At Random*): a ausência não carrega informação sobre o alvo
- O dataset é grande o suficiente para perder linhas sem prejudicar o treino

### Quando evitar

- Muitos nulos: remover 30% das linhas perde sinal e distorce distribuições
- Nulos não-aleatórios: se usuários de alta renda tendem a não preencher `income`, removê-los cria viés — o modelo aprende com uma amostra que não representa a população real

### Trade-off

| Prós | Contras |
|------|---------|
| Zero complexidade | Perde dados — reduz tamanho do treino |
| Sem risco de introduzir ruído | Pode criar viés se a ausência não for aleatória |
| Rápido de implementar | Problemas em produção se nulos aparecerem em novas predições |

---

## 3. Técnica 2 — Imputação por Média / Mediana / Moda

### Ideia central

Substitui o `NaN` por um **resumo estatístico** calculado nos dados de treino. Simples e interpretável.

> **Regra de ouro:** calcule a estatística **sempre no conjunto de treino** e aplique no teste. Usar o teste para calcular a média seria *data leakage* — o modelo "veria" o futuro.

---

### Média (*mean*)

```python
X_train_m['age'] = X_train_m['age'].fillna(np.mean(X_train_m['age']))
X_test_m['age']  = X_test_m['age'].fillna(np.mean(X_train_m['age']))  # usa média do treino
```

**Quando usar:** variáveis numéricas contínuas com distribuição aproximadamente simétrica (sem outliers fortes).

**Problema:** sensível a outliers. Se `income` tem alguns valores extremos, a média sobe — e todos os nulos viram esse valor distorcido.

---

### Mediana (*median*)

```python
m_df['age'] = df['age'].fillna(np.median(m_df['age']))
```

**Quando usar:** variáveis numéricas com distribuição assimétrica ou outliers (salários, preços, idade em datasets com extremos).

**Por que é mais robusta:** a mediana é o valor do meio da distribuição ordenada — outliers não a movem tanto quanto a média.

| Distribuição | Escolha |
|---|---|
| Simétrica, sem outliers | Média |
| Assimétrica ou com outliers | Mediana |

---

### Moda (*mode*)

```python
m_df['age'] = m_df['age'].fillna(stats.mode(m_df['age'])[0][0])
```

**Quando usar:** variáveis **categóricas** (país, plano, gênero) — média e mediana não fazem sentido para texto ou classes discretas.

`stats.mode()` retorna um objeto com o valor mais frequente em `[0][0]`.

---

### Limitação geral da média/mediana/moda

Todos os nulos de uma coluna viram **o mesmo valor**. Isso:

- Comprime a variância da coluna artificialmente
- Pode esconder padrões reais por trás dos nulos

Para datasets grandes com poucos nulos, funciona bem. Para mais nulos ou dados com estrutura complexa, as técnicas avançadas abaixo são melhores.

---

## 4. Técnica 3 — Imputação Múltipla por Regressão (`IterativeImputer`)

### Ideia central

Em vez de um único valor fixo, o `IterativeImputer` usa **um modelo de ML** para prever o valor ausente com base nas outras colunas. Itera várias vezes, refinando cada imputação.

Isso é chamado de **MICE** (*Multiple Imputation by Chained Equations*) na literatura estatística.

```python
Imp = IterativeImputer(max_iter=10, random_state=0)
Imp.fit(X_train_r)

X_train_r = Imp.transform(X_train_r)
X_test_r  = Imp.transform(X_test_r)
```

### Como funciona internamente

1. Preenche todos os nulos com a média inicialmente
2. Para cada coluna com nulo: usa as **outras colunas como features** e treina um modelo (default: `BayesianRidge`, uma regressão linear regularizada) para prever o valor ausente
3. Repete o processo `max_iter` vezes — a cada rodada, as imputações ficam mais consistentes entre si

```
Rodada 1: imputa age   → usa days_on_platform, income (valores da rodada anterior)
           imputa income → usa age (já imputado), days_on_platform
Rodada 2: reimputa age  → agora income está melhor imputado → age melhora também
...
Rodada 10: convergência
```

### Parâmetros principais

| Parâmetro | O que faz | Valor típico |
|-----------|-----------|-------------|
| `estimator` | Modelo usado para prever o nulo | `BayesianRidge` (default) ou `RandomForestRegressor` |
| `max_iter` | Número de rodadas de imputação | 10 |
| `random_state` | Reprodutibilidade | 0 |
| `add_indicator` | Cria coluna binária indicando onde foi imputado | `True` (recomendado) |
| `missing_values` | Tipo do valor nulo a imputar | `np.nan` (default) |

### `add_indicator` — por que é importante

Quando `add_indicator=True`, o imputer adiciona uma coluna extra `0/1` por feature imputada:

| age | age_was_imputed |
|-----|----------------|
| 35  | 0 |
| 28  | 1 (era NaN) |

**Por quê isso importa:** a *ausência* de um valor pode carregar informação (ex.: usuários que não informam renda podem ter padrão de compra diferente). Guardar onde foi imputado permite ao modelo downstream capturar esse padrão.

### Estimadores disponíveis

| Estimador | Descrição | Quando usar |
|-----------|-----------|-------------|
| `BayesianRidge` | Regressão linear regularizada (default) | Relações lineares entre features; rápido |
| `RandomForestRegressor` | Random Forest — captura relações não-lineares | Dados complexos; mais lento; equivalente ao `missForest` do R |

### Após a transformação — restaurar DataFrame

```python
X_train_r = pd.DataFrame(X_train_r)
X_train_r.columns = X_train_r.columns  # restaurar nomes das colunas
```

O `transform` retorna um array NumPy — é preciso converter de volta para DataFrame para manter nomes de colunas legíveis.

---

## 5. Técnica 4 — Imputação por Vizinhos Mais Próximos (`KNNImputer`)

### Ideia central

Para cada valor ausente, o `KNNImputer` encontra os **K registros mais similares** (vizinhos mais próximos) que têm valor naquele campo — e usa a média (ou média ponderada) desses vizinhos como imputação.

```python
imputer = KNNImputer(n_neighbors=5, weights="uniform")
imputer.fit(X_train_r)
X_train_k = imputer.transform(X_train_r)
X_test_k  = imputer.transform(X_test_r)
```

### Como funciona internamente

1. Para o registro com `age = NaN`, calcula a distância (Euclidiana por default) para todos os outros registros usando as colunas disponíveis (`days_on_platform`, `income`)
2. Encontra os 5 vizinhos com menor distância **que têm `age` preenchido**
3. Imputa a média dos `age` desses 5 vizinhos

### Parâmetros principais

| Parâmetro | O que faz | Valor típico |
|-----------|-----------|-------------|
| `n_neighbors` | Número de vizinhos usados | 5 (default) |
| `weights` | Como ponderar os vizinhos | `"uniform"` (igual) ou `"distance"` (mais perto = mais peso) |
| `metric` | Métrica de distância | `"nan_euclidean"` (default, ignora NaN no cálculo) |
| `add_indicator` | Coluna indicadora de imputação | `False` (default) |
| `missing_values` | Tipo do valor nulo | `np.nan` |

### `weights` — uniform vs. distance

| `weights` | Comportamento |
|-----------|--------------|
| `"uniform"` | Todos os K vizinhos têm peso igual — média simples |
| `"distance"` | Vizinhos mais próximos têm mais influência — média ponderada pelo inverso da distância |
| `callable` | Função customizada que recebe array de distâncias e retorna pesos |

**Regra prática:** `"distance"` tende a ser mais preciso quando os vizinhos mais próximos são de fato muito similares. `"uniform"` é mais estável com K pequeno.

### `n_neighbors` — overfitting vs. underfitting

| K pequeno (ex.: 1–3) | K grande (ex.: 20+) |
|---------------------|---------------------|
| Imputação mais específica | Imputação mais suavizada |
| Risco de overfitting (depende muito de poucos pontos) | Perde precisão local |

**Default K=5** é um bom ponto de partida para a maioria dos datasets.

---

## 6. Comparação entre técnicas

O notebook treina um `RandomForestRegressor` com cada conjunto de dados imputado e compara o **MAE** (*Mean Absolute Error* — erro médio absoluto) nas predições de `lifetime_value`:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print('Drop Null MAE Score: %.3f'       % mean_absolute_error(y_test_d, pred_dropna))
print('Mean Impute MAE Score: %.3f'     % mean_absolute_error(y_test_m, pred_m))
print('Regression MAE Score: %.3f'      % mean_absolute_error(y_test_r, pred_r))
print('Nearest Neighbor MAE Score: %.3f'% mean_absolute_error(y_test_k, pred_k))
```

**MAE** mede, em média, o quanto a predição erra em unidades do alvo. Menor = melhor.

### Resumo comparativo das técnicas

| Técnica | Complexidade | Quando usar | Risco principal |
|---------|-------------|-------------|-----------------|
| **Drop Null** | Muito baixa | Poucos nulos aleatórios | Perda de dados e viés |
| **Média** | Baixa | Muitos nulos, distribuição simétrica, pouco tempo | Comprime variância; sensível a outliers |
| **Mediana** | Baixa | Muitos nulos, distribuição assimétrica ou outliers | Comprime variância |
| **Moda** | Baixa | Variáveis categóricas | Comprime variância |
| **IterativeImputer** | Alta | Relações entre features importam; aceita demora de treino | Mais lento; risco de overfitting se `max_iter` alto |
| **KNNImputer** | Média-alta | Similaridade entre registros é informativa | Lento em datasets grandes; sensível à escala das features |

---

## 7. Boas práticas gerais

1. **Sempre use os dados de treino** para calcular estatísticas (média, mediana, vizinhos) e depois aplique no teste — nunca o contrário (data leakage).

2. **`df.copy()`** antes de imputar: preserve o DataFrame original para poder comparar técnicas sem re-carregar o CSV.

3. **`add_indicator=True`** em `IterativeImputer` / `KNNImputer` quando suspeitar que o padrão de ausência é informativo.

4. **Escale as features** antes do KNNImputer — distância Euclidiana é sensível à escala. Uma coluna com valores em milhares domina a distância sobre uma em decimais.

5. **Compare no downstream** (como no notebook): a melhor técnica de imputação é aquela que produz **melhor MAE/AUC no modelo final**, não a mais sofisticada em teoria.

6. **Documente** qual técnica foi usada e por quê — em produção, o pipeline precisa repetir exatamente a mesma imputação do treino.

---

## Suas notas

- 
- 

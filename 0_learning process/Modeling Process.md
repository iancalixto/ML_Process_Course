# Modeling Process

Notas do curso **Machine Learning Process** — passo a passo do **processo de modelagem**, da definição do problema até colocar o modelo em produção.

> Complementa: [Intro to Machine Learning.md](./Intro%20to%20Machine%20Learning.md)

---

## Visão geral do pipeline

```
  Enquadramento → Coleta → Pré-processamento → EDA → Feature engineering
       → Modelagem → Validação → Avaliação → Produtização
            ↑___________________________________________|
                         (iterar)
```

---

## 1. Enquadramento do problema (*problem framing*)

Para aplicar ML de forma eficaz, o primeiro passo é **entender e definir o problema** — não pular direto para dados e modelos.

### Entender o problema

Muitos pedidos chegam vagos. O trabalho do enquadramento é **transformar um problema vago em um problema definido**, com escopo, métrica e critério de sucesso.

**Exemplo:** *“O pipeline quebrou”* ou *“precisamos consertar um bug no pipeline”* ainda é amplo. Perguntas de enquadramento levam a algo acionável, por exemplo:

- Qual etapa do pipeline falha e com que frequência?
- Isso bloqueia treino, inferência ou relatórios?
- A correção é engenharia de dados, regra de negócio ou realmente ML?

> ML só entra quando o problema **definido** se beneficia de predição, segmentação ou automação em escala — não quando o problema ainda é “algo está errado”.

---

### Perguntas essenciais

Responda com clareza (por escrito, com o time):

| Pergunta | Por que importa |
|----------|-----------------|
| **Quem é o usuário final?** | Produto, ops, analista, cliente B2B? Define UX da solução e como medir sucesso. |
| **Qual problema estamos tentando resolver?** | Uma frase específica — não “usar IA”, mas “reduzir churn D30 em usuários free”. |
| **Qual é o impacto?** | Receita, custo, risco, tempo da equipe, experiência do usuário — quantificar se possível. |
| **Qual é a urgência?** | Prazo, janela de campanha, incidente em produção — prioriza simples vs. ML completo. |

---

### Escala, restrições e dependências

Antes de escolher algoritmo, mapeie o **contexto real**:

**Escala**

- Quantos usuários, eventos, linhas por dia?
- Batch (diário) ou tempo real (milissegundos)?
- Crescimento esperado nos próximos 12 meses?

**Restrições (*constraints*)**

- Orçamento, prazo, equipe (ML, eng, produto)
- Privacidade / LGPD, dados que não podem sair de certas regiões
- Interpretabilidade obrigatória vs. máxima acurácia
- Infra existente (cloud, feature store, API)

**Dependências**

- De quais times ou sistemas dependemos? (dados, deploy, legal)
- Quais pipelines, tabelas ou APIs precisam existir antes do modelo?
- O que bloqueia o projeto se não for entregue primeiro?

---

### Como levantar contexto (antes do notebook)

| Atividade | Objetivo |
|-----------|----------|
| **Entrevistar stakeholders** | Alinhar problema, impacto, urgência e definição de “sucesso” com produto, negócio, engenharia |
| **Ler documentação** | PRDs, specs, runbooks, incidentes passados, definições de métricas no warehouse |
| **Análise exploratória inicial** | Validar se os dados existem, se há sinal e se a pergunta faz sentido — *exploratory analysis* ligada ao enquadramento, não só EDA “de modelo” |

Documente hipóteses e decisões (mesmo em bullet points) para não refazer o enquadramento a cada sprint.

---

### Problema de qualidade (bem definido)

Um bom enquadramento é:

| Critério | O que significa |
|----------|-----------------|
| **Quantitativo** | Métricas numéricas de sucesso (%, R$, tempo, volume) — não só “melhorar” |
| **Específico** | Escopo claro: quem, o quê, quando, onde — sem ambiguidade |
| **Focado no usuário** | Benefício para quem usa o produto ou para o stakeholder que decide |
| **Com restrições** | Limites explícitos — muitas vezes **baseados em tempo** (prazo, SLA, janela de dados históricos) |

---

### Impacto de negócio (*business impact*)

O objetivo final é **gerar valor de negócio** — não só um modelo com boa métrica técnica.

**Duas camadas de score**

| Tipo | O que mede | Exemplos |
|------|------------|----------|
| **Accuracy score** (métrica do modelo) | Performance técnica offline/online | AUC, F1, RMSE, precisão@k |
| **Business score** (métrica de negócio) | Impacto real no KPI | Retenção D30, receita incremental, custo evitado, tempo economizado |

**Como pensar o “business model” do projeto**

Não é necessariamente um modelo financeiro completo no dia 1, mas sim uma conta explícita:

1. **Baseline** — o que acontece *sem* a solução nova (regra atual, processo manual, modelo legado)?  
2. **Lift esperado** — se o modelo acerta X% a mais, quantos usuários/reais isso move?  
3. **Conversão em valor** — clique → compra → margem; churn evitado → LTV preservado.  
4. **Custo do projeto** — tempo de ML, eng, infra, manutenção vs. ganho.

> Queremos garantir que o modelo **cria valor de negócio**, não apenas um relatório com AUC alto.

**Validação do problema**

- Revisar **definições** (o que é churn? conversão? janela de 7 ou 30 dias?)  
- Validar com stakeholders se a métrica de sucesso é a certa  
- **Buscar soluções diferentes** antes de fixar em um algoritmo: regra, dashboard, heurística, ML supervisionado, ML não supervisionado

---

### Escolha da solução (antes de codar o modelo)

Compare alternativas com **prós e contras** e matriz **esforço × impacto**.

**Perguntas centrais**

| Pergunta | Guia rápido |
|----------|-------------|
| **Que tipo de algoritmo dá o melhor resultado?** | Depende dos dados e do alvo — testar baseline simples primeiro, depois candidatos mais complexos |
| **Supervisionado ou não supervisionado?** | Temos rótulo/alvo histórico? → supervisionado. Só queremos descobrir grupos/padrões? → não supervisionado |
| **Classificação ou regressão?** | Alvo categórico (sim/não, classe) → classificação. Alvo contínuo (valor, tempo, contagem) → regressão |

**Classificação vs. regressão — prós e contras**

| | **Classificação** | **Regressão** |
|---|-------------------|---------------|
| **Prós** | Decisão clara (aprovar/reprovar, churn sim/não); fácil de comunicar ao negócio | Estima magnitude (quanto vale, quanto tempo); útil para priorização fina |
| **Contras** | Perde granularidade se o negócio precisa de “quanto” | Erros em escala podem distorcer ROI; stakeholders podem querer threshold mesmo assim |

**Esforço × impacto**

```
        impacto alto
             │
   quick     │    strategic
   wins      │    (ML full)
             │
        ─────┼───── esforço alto
             │
   evitar     │    reconsiderar
             │
        impacto baixo
```

Priorize soluções no quadrante **alto impacto, baixo esforço** antes de um pipeline ML longo.

---

### Entender os dados (*data audit*)

Antes de modelar, audite o que existe.

| Pergunta | Como responder |
|----------|----------------|
| **Que dados temos?** | Inventário de tabelas, eventos, período, granularidade (usuário, sessão, item) |
| **Os dados estão aceitáveis no estado atual?** | Qualidade, nulos, atraso (latência), consistência de definições |
| **Há features e sinais suficientes para um modelo de alta performance?** | Correlação com o alvo, variância, cobertura por segmento |
| **Precisamos coletar mais dados?** | Lacunas de tracking, histórico curto, labels ausentes |

**Ferramentas para obter respostas**

- **Consultas SQL** — volumes, null rate, distribuição por cohort, amostras de casos extremos  
- **Matrizes de correlação** — relação entre features e alvo; multicolinearidade entre features  

Documente limitações: “dados aceitáveis com ressalvas” vs. “bloqueado até coletar X”.

---

### Checklists consolidados

#### Entendimento do problema

- [ ] Qual problema estamos resolvendo **para eles** (usuário / stakeholder)?  
- [ ] O que é **sucesso** para este problema? (métrica + meta)  
- [ ] Qual o **impacto de negócio**?  
- [ ] Qual a **urgência**?  
- [ ] Qual a **escala**?  
- [ ] Qual o **nível de complexidade**?  
- [ ] Quais **restrições** (incl. tempo)?  
- [ ] Quais **dependências**?  
- [ ] Problema **quantitativo**, **específico** e **focado no usuário**?

#### Solução

- [ ] Tipo de algoritmo / abordagem para melhor resultado (com baseline)  
- [ ] **Supervisionado** ou **não supervisionado** — justificado  
- [ ] **Classificação** ou **regressão** — prós e contras documentados  
- [ ] Matriz **esforço × impacto** para alternativas  
- [ ] **Accuracy score** e **business score** definidos  
- [ ] Plano de validação e comparação de soluções diferentes  

#### Coleta e qualidade dos dados

- [ ] Que **dados** endereçam o problema (fontes, período)?  
- [ ] Dados **aceitáveis** no estado atual?  
- [ ] **Features e sinais** suficientes para performance alta?  
- [ ] Precisa **coletar mais dados** ou melhorar tracking?  
- [ ] Auditoria feita (SQL, correlações)?  

### Suas notas

- 
- 

---

## 2. Coleta de dados

Após o *data audit* do enquadramento, execute e documente a coleta.

> Detalhes das fontes (SQL, NoSQL, APIs, scraping, surveys): [Data Collection.md](./Data%20Collection.md)

| Pergunta | Ação |
|----------|------|
| Que dados endereçam o problema? | Listar tabelas, pipelines, APIs |
| Estado aceitável? | Critérios de qualidade mínima acordados |
| Sinais suficientes? | Gap analysis de features vs. alvo |
| Coletar mais? | Tickets para eng/analytics, novos eventos no app |

*Suas notas:*

- 
- 

---

## 3. Pré-processamento

*Tratar nulos, outliers, tipos, consistência entre fontes.*

> Guia completo: [Data Processing.ipynb](./Data%20Processing.ipynb)

*Suas notas:*

- 
- 

---

## 4. Análise exploratória (EDA)

*Distribuições, correlações, vazamento de dados, sinal vs. ruído.*

*Suas notas:*

- 
- 

---

## 5. Feature engineering

*Features candidatas, origem, disponibilidade em produção.*

*Suas notas:*

- 
- 

---

## 6. Modelagem

*Baselines, algoritmos testados, hiperparâmetros.*

*Suas notas:*

- 
- 

---

## 7. Validação cruzada

*K-fold vs. validação temporal; splits treino / validação / teste.*

*Suas notas:*

- 
- 

---

## 8. Avaliação

*Métricas offline, métricas de negócio, testes online (A/B).*

*Suas notas:*

- 
- 

---

## 9. Produtização

*API, monitoramento, retreino, fallback, launch.*

*Suas notas:*

- 
- 

---

## Plano de projeto (template)

| Item | Definição |
|------|-----------|
| Problema / KPI | |
| Intervalo de dados | |
| Volume de dados | |
| Features | |
| Modelos a testar | |
| Métrica do modelo vencedor | |
| Como servir em produção | |

---

## Experimentos e decisões

| Data | Experimento | Resultado | Próximo passo |
|------|-------------|-----------|---------------|
| | | | |
| | | | |

---

## Referências e links

- 
- 

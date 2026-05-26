# Intro to Machine Learning

Notas do curso **Machine Learning Process** — conceitos, exemplos e fluxos de trabalho em analytics e ML.

---

## Hierarquia de necessidades em Analytics

Inspirada na pirâmide de Maslow, a **hierarquia de necessidades em analytics** mostra que cada etapa depende da anterior. Você só sobe de nível quando a base está sólida — não adianta treinar modelos de ML sem dados confiáveis e KPIs bem definidos.

```
                    ┌─────────────────────────────┐
                    │  5. Otimizar e prever     │
                    │     com Machine Learning    │
                    └──────────────┬──────────────┘
                 ┌─────────────────┴─────────────────┐
                 │  4. Analisar KPIs e gerar insights  │
                 └─────────────────┬─────────────────┘
              ┌────────────────────┴────────────────────┐
              │  3. Definir KPIs e acompanhar evolução  │
              └────────────────────┬────────────────────┘
           ┌───────────────────────┴───────────────────────┐
           │           2. Limpar os dados                  │
           └───────────────────────┬───────────────────────┘
        ┌──────────────────────────┴──────────────────────────┐
        │              1. Coletar os dados                    │
        └─────────────────────────────────────────────────────┘
```

### Níveis (de baixo para cima)

**1. Coletar os dados** *(base)*  
Sem dados, não há analytics. Aqui entram fontes (planilhas, APIs, bancos, sensores), pipelines de ingestão e garantia de que a informação chega de forma repetível e documentada.

**2. Limpar os dados**  
Dados brutos raramente estão prontos para análise. Tratamos valores ausentes, duplicatas, tipos incorretos, outliers e inconsistências. Qualidade aqui evita conclusões erradas em todos os níveis acima.

**3. Definir KPIs e acompanhar a evolução**  
KPIs (Key Performance Indicators) traduzem objetivos de negócio em métricas mensuráveis. Definimos o que importa, como medir e monitoramos tendências ao longo do tempo — dashboards e relatórios vivem principalmente neste nível.

**4. Analisar KPIs e gerar insights**  
Com dados limpos e KPIs estáveis, exploramos padrões, comparamos períodos, segmentamos públicos e respondemos *por quê* algo mudou. O foco passa de *o que aconteceu* para *o que isso significa*.

**5. Otimizar e prever com Machine Learning** *(topo)*  
No topo da pirâmide usamos modelos para prever resultados, recomendar ações ou automatizar decisões. ML só agrega valor quando os quatro níveis inferiores estão maduros — caso contrário, o modelo aprende ruído ou métricas erradas.

### Ideia central

Quanto mais alto na pirâmide, maior o impacto potencial — e maior a dependência da base. Pular etapas (por exemplo, ir direto para ML sem limpeza ou sem KPIs claros) costuma gerar projetos caros que não sustentam decisões no dia a dia.

---

## Quando usar Machine Learning?

**ML existe para economizar tempo** — automatizar ou acelerar trabalho que seria lento, repetitivo ou caro se feito só por pessoas. Não é um fim em si; precisa gerar **valor de negócio** mensurável.

### Escada de complexidade (do simples ao ML)

Comece pelo mais simples que resolve o problema. Só suba de degrau quando o anterior não for suficiente:

```
  Regras simples (heurísticas simples)
           ↓
  Regras mais elaboradas (heurísticas complexas)
           ↓
  Machine Learning
```

| Abordagem | Quando considerar |
|-----------|-------------------|
| **Heurísticas simples** | Poucos dados, regras óbvias (“se X, então Y”), problema bem definido |
| **Heurísticas complexas** | Muitas condições e exceções, mas ainda explicáveis em linguagem de negócio |
| **Machine Learning** | Muito dado, muitas variáveis, padrões difíceis de codificar à mão |

### Antes de escolher ML, verifique

1. **Tamanho dos dados** — ML costuma exigir volume suficiente para generalizar; com poucos registros, regras ou estatística simples podem ser melhores.

2. **Variáveis explicativas (sinais)** — O modelo só aprende o que está nos dados. Liste features candidatas: estão disponíveis, confiáveis e no momento certo (sem vazamento de informação do futuro)?

3. **Poder explicativo dos dados** — Avalie se os sinais realmente se relacionam com o alvo (correlação, EDA, testes simples). Se não houver sinal, ML não “inventa” insight.

4. **Valor de negócio** — Qual decisão ou processo melhora? Quanto tempo/dinheiro se economiza? Como medir sucesso depois do deploy?

5. **Teste uma solução sem ML primeiro** — Baseline com regras, médias, thresholds ou relatórios manuais. Se já atende o negócio, ML pode ser overkill.

### “Falta de dados” depende do problema

Não existe um número mágico de linhas para todo projeto. A quantidade necessária varia com:

- **Complexidade do problema** — classificar spam é diferente de prever falha em equipamento industrial raro
- **Número de variáveis e classes** — mais features e categorias costumam exigir mais exemplos
- **Ruído e desbalanceamento** — dados sujos ou classes raras pedem mais volume ou técnicas específicas
- **Margem de erro aceitável** — decisões de alto risco exigem mais dados e validação

Ou seja: “dados insuficientes” é relativo ao **tipo de problema**, não só ao tamanho da tabela.

### Restrições de tempo (dois lados)

| Lado | O que considerar |
|------|------------------|
| **Prazo do negócio** | O resultado precisa estar pronto *quando*? Um modelo em 6 meses pode chegar tarde demais. |
| **Tempo para construir ML** | Desenvolver um sistema de ML leva tempo: coleta, limpeza, features, treino, validação, deploy, monitoramento. Não é “só treinar um modelo”. |

Se o prazo é curto, uma **solução simples entregue cedo** quase sempre vence um projeto de ML ainda em laboratório.

### Alerta: ainda não testou o simples

Um sinal clássico de que ML **não** deveria ser o próximo passo:

> **Ainda não tentaram uma solução simples.**

Sem baseline (regra, média, planilha, dashboard manual), não dá para saber se ML vale o investimento. Ir direto para modelo costuma misturar problema mal definido com complexidade desnecessária.

### 80% do resultado vs. os últimos 20% (solução simples → ML)

Muitas vezes a progressão é assim:

```
  Solução simples  ──────────►  ~80% do valor / da qualidade desejada
         │
         │  (muito esforço extra)
         ▼
  Modelo de ML     ──────────►  dos 80% aos 100% (ganhos marginais)
```

- A **solução simples** (heurística, relatório, regra de negócio) entrega a maior parte do benefício com pouco custo e rápido de manter.
- O **ML** entra para refinar: capturar os casos difíceis, ganhar alguns pontos de acurácia ou automatizar em escala — o salto de **80% → 100%** do output.

Pergunta honesta antes do ML: *os últimos 20% justificam semanas/meses de engenharia, infra e manutenção?* Se 80% já resolve o KPI, pare aí.

### Sinais de que ML faz sentido

Use ML quando **vários** destes pontos forem verdadeiros:

- **Muito dado** (volume e/ou variedade que humanos não processam bem)
- **Problema que consome muito tempo** de forma repetitiva
- **Caro resolver só com pessoas** (escala, especialistas escassos, 24/7)
- **Heurísticas já não escalam** — regras ficaram frágeis, cheias de exceções ou imprecisas
- Há **retorno claro** (receita, custo, risco, experiência do usuário)

### Resumo prático

> **ML para ganhar tempo, não para impressionar.**  
> “Falta de dados” depende do problema → respeite prazos do negócio e o tempo de *construir* ML → **teste o simples antes** (80% do valor) → ML só se os últimos 20% valem o custo.

---

## Exemplo: otimizar recomendações para subir retenção

Cenário típico: *“Podemos aumentar o KPI de **retenção** se otimizarmos o mecanismo atual de **recomendação**.”*

Antes de assumir que ML é o caminho, lembre: **modelos exigem muito trabalho**. Vale a pena só com problema bem definido e plano claro — definição errada do problema pode custar **meses de esforço desperdiçado**.

### 1. Definir o problema com clareza (crítico)

Responda por escrito, com o negócio:

- O que é **retenção** aqui? (D7? D30? assinatura ativa? churn invertido?)
- O que significa **otimizar recomendação**? (cliques, tempo na sessão, compra, retorno no app?)
- Qual hipótese liga recomendação → retenção? Como medir **antes/depois**?
- O que **não** é escopo (evita scope creep)?

> Definição incorreta → equipe treina o modelo certo para a pergunta errada.

### 2. Análise de correlação e abordagem baseada em features

Antes de modelar:

- **Análise de correlação** entre sinais candidatos e o alvo (retenção / proxy)
- Entender multicolinearidade e variáveis fracas
- **Abordagem orientada a features**: listar o que o usuário fez, viu, comprou, quando, em qual dispositivo, etc.
- Priorizar features com **poder explicativo** e disponíveis em produção no momento da predição

### 3. Plano de projeto (documento vivo)

| Item | O que definir |
|------|----------------|
| **Intervalos de dados** | De quando até quando? treino / validação / teste temporal |
| **Volume de dados** | Quantos usuários, eventos, itens — suficiente para o problema? |
| **Features a incluir** | Lista, origem, latência, tratamento de missing |
| **Modelos a testar** | Baselines + candidatos (ex.: regras, matrix factorization, gradient boosting) |
| **Como colocar em produção** | batch vs. tempo real, SLA, fallback se modelo falhar |
| **Critério de “modelo vencedor”** | métrica offline + métrica de negócio (retenção, não só AUC) |

### 4. Ciclo de experimentação → modelo vencedor

```
  Explorar dados → treinar candidatos → comparar métricas → iterar
                              ↑______________________________|
```

- Testar várias combinações de features e algoritmos
- Validar com hold-out ou validação temporal (recomendação costuma ser **série temporal**)
- Só parar quando houver um **modelo vencedor** claro vs. baseline atual (recomendação legada)

### 5. Produção: da predição rápida ao lançamento

Fluxo típico após escolher o modelo:

1. **Colocar em produção (engineering)**  
   - Desenvolver uma **API REST** (ou serviço equivalente) que devolve predições/recomendações **rápido** o suficiente para o app ou produto  
   - Integrar na solução existente (substituir ou A/B testar o mecanismo atual)

2. **Testar performance do modelo**  
   - Testes offline já feitos; em seguida **teste online** (shadow mode, A/B, canary)  
   - Monitorar latência, erros, drift de dados e impacto no KPI de retenção

3. **Lançamento (launch) em produção**  
   - Rollout gradual, rollback planejado, documentação para ops e produto  
   - Acompanhar retenção e métricas secundárias após o go-live

### Visão em uma linha

> Hipótese de negócio (retenção ↑ via recomendação) → **problema bem definido** → correlação + features → **plano** → experimentos até modelo vencedor → **API rápida** no app → testes → **launch** em produção.

---

## Aprendizado supervisionado vs. não supervisionado

| | **Supervisionado** | **Não supervisionado** |
|---|-------------------|------------------------|
| Objetivo | Prever um **resultado específico** | Descobrir **estrutura** nos dados sem rótulo |
| “Gabarito” | Sim — humanos fornecem **dados rotulados** | Não — o algoritmo **cria os rótulos/grupos** sozinho |
| Analogia | Máquina com **caderno de respostas** | Máquina **sem** caderno de respostas |

---

### Aprendizado supervisionado

No supervisionado, queremos **prever algo conhecido** — classificar ou estimar um valor que já sabemos como medir em exemplos passados.

**Como funciona**

- Humanos fornecem **dados rotulados** (cada exemplo vem com a “resposta certa”).
- O modelo aprende a relação entre **entradas (features)** e **saída (rótulo)**.
- Depois usa isso para prever em dados novos.

**Analogia: máquina com gabarito**

É como estudar para prova com o **caderno de respostas**: você vê a pergunta, tenta responder e confere se acertou.

**Exemplo: é mamão ou não?**

Imagine classificar frutas. Para cada mamão (ou não mamão) você mede:

- maciez  
- comprimento  
- massa  
- teor de açúcar  

E alguém já marcou: **“isto é mamão”** ou **“não é mamão”**. O ML aprende com esse **gabarito** — features na entrada, rótulo na saída. Em produção, o modelo prevê a classe de uma fruta nova só pelas medições.

**Outros casos supervisionados**

- Prever churn (sim/não)  
- Estimar preço de imóvel (regressão)  
- Detectar spam (rótulo: spam / não spam)

---

### Aprendizado não supervisionado

Aqui **não há resposta certa** no treino. O algoritmo explora os dados e **descobre padrões, grupos ou representações** por conta própria.

**Como funciona**

- Só temos features — **sem rótulo** humano para cada linha.
- O modelo agrupa, reduz dimensionalidade ou aprende uma estrutura latente.
- Os “rótulos” (segmentos, clusters) **emergem** dos dados.

**Analogia: sem gabarito**

É como pedir à máquina: *“olha estes clientes e me diz que tipos de público existem”* — sem dizer antes quantos grupos nem quem é quem.

**Exemplo: segmentos de audiência**

Você passa dados de comportamento (idade, gênero, interesses, renda, etc.) **sem** dizer o segmento. O algoritmo pode descobrir grupos como:

- homens jovens, amam esportes  
- público mais velho, orientado à família  
- perfil artístico / cultural  
- mulheres jovens, alta renda  

Esses clusters **não foram definidos manualmente** — o ML sugeriu divisões úteis para marketing, produto ou conteúdo.

**Outro tipo: generativo (muitas vezes não supervisionado)**

Em tarefas **generativas** (criar texto, imagem, áudio), o modelo aprende a **distribuição** dos dados — por exemplo, ao gerar frases, aprende padrões da linguagem sem precisar de um rótulo “certo” para cada palavra nova. Chatbots e modelos de linguagem entram nessa família (com nuances: alguns usam fine-tuning supervisionado depois).

---

### Resumo rápido

| Pergunta | Supervisionado | Não supervisionado |
|----------|----------------|----------------------|
| Temos a resposta no treino? | **Sim** (rótulos) | **Não** |
| O que o modelo faz? | **Prevê** o rótulo/alvo | **Descobre** grupos ou estrutura |
| Exemplo do curso | Mamão: maciez, massa, açúcar → é mamão? | Segmentos de audiência sem rótulo prévio |
| Risco se dados forem fracos | Rótulos errados ou desbalanceados | Clusters difíceis de interpretar no negócio |

---

## Regressão vs. classificação

Ambos são tipos de **aprendizado supervisionado**: temos um alvo (o que queremos prever) e dados históricos com resposta conhecida. A diferença está no **tipo de variável alvo**.

| | **Regressão** | **Classificação** |
|---|---------------|-------------------|
| Alvo | Variável **contínua** (número em escala) | Variável **categórica** (classe / rótulo) |
| Pergunta típica | *“Quanto?”* | *“Qual categoria?”* |
| Saída do modelo | Valor numérico (ex.: R$ 1.240,50) | Classe ou probabilidade por classe (ex.: “churn”, 78%) |

---

### Regressão — prever variáveis contínuas

**Objetivo:** estimar um **número** (valor, tempo, quantidade, score).

**Exemplos do mundo real**

| Domínio | O que prever |
|---------|----------------|
| **Uber** | **ETA** (tempo estimado de chegada) do motorista — minutos até o destino |
| **Airbnb** | **Valor** de uma reserva / preço sugerido para o imóvel |
| **Redes sociais (ads)** | **Valor** de lances ou performance de pedidos de anúncio (CPM, CPC, conversão em $) |
| **Negócio geral** | **Customer Lifetime Value (CLV)** — quanto um cliente vale ao longo do tempo |
| **Vendas** | **Receita**, **volume de vendas**, **número de vendas** em um período |

O modelo aprende: *dadas estas features (distância, bairro, histórico, etc.), qual número esperamos?*

**Métricas comuns:** erro médio (MAE), erro quadrático (RMSE), R².

---

### Classificação — prever categorias

**Objetivo:** atribuir cada exemplo a uma **classe** (sim/não, positivo/negativo/neutro, segmento A/B/C).

**Exemplos do mundo real**

| Domínio | O que classificar |
|---------|-------------------|
| **Retenção** | Usuário **vai permanecer** ou **vai churnar**? (sim/não) |
| **Conversão** | Visitante **converteu** (comprou, assinou) ou não? |
| **E-mail marketing** | E-mail foi **aberto** ou não? |
| **Sentimento** | Feedback do usuário: **positivo**, **negativo** ou **neutro** |

O modelo aprende: *dado este perfil/comportamento, qual rótulo discreto encaixa melhor?*

**Métricas comuns:** acurácia, precisão, recall, F1, AUC-ROC (especialmente em classes desbalanceadas).

---

### Product analytics — onde cada um aparece

Em **analytics de produto**, regressão e classificação respondem perguntas diferentes:

**Regressão (quanto / quanto tempo)**

- Prever **CLV** para priorizar investimento em aquisição  
- Estimar **receita** ou **número de eventos** (sessões, compras) no próximo mês  
- Prever **tempo até conclusão** de onboarding ou tempo na tela  

**Classificação (qual grupo / qual decisão)**

- **Retenção:** usuário de risco de churn nos próximos 7 dias?  
- **Conversão:** trial vira pagante?  
- **Engajamento:** abriu push / e-mail da campanha?  
- **Sentimento:** NPS aberto ou review → satisfeito vs. insatisfeito  

```
  Product analytics
        │
        ├── Regressão     → KPIs numéricos (CLV, revenue, counts, ETA…)
        │
        └── Classificação → KPIs categóricos (retido?, converteu?, abriu?, sentimento…)
```

---

### Como escolher na prática

1. Olhe o **KPI de negócio**: é um número contínuo ou uma categoria?  
2. Se o time fala “**probabilidade de churn**”, ainda é **classificação** (classe churn / não churn); a probabilidade é só a forma de expressar a confiança.  
3. Muitos problemas de produto começam como **classificação** (ação binária) e depois evoluem para **regressão** (ex.: prever *quanto* de receita um segmento gera).

### Resumo

> **Regressão** = prever **quanto** (CLV, receita, ETA, preço).  
> **Classificação** = prever **qual categoria** (retenção, conversão, abertura de e-mail, sentimento).

---

## Product analytics — fluxo de trabalho

Em analytics de produto, o trabalho raramente começa no modelo. Segue-se um pipeline do dado bruto até **hipóteses testáveis** que o negócio pode agir.

```
  Pré-processamento → EDA → Feature engineering → Modelagem → Hipóteses / insights
        ↑___________________________________________________________________|
                         (iterar conforme aprende)
```

---

### 1. Pré-processamento de dados (*data preprocessing*)

Preparar dados confiáveis **antes** de analisar ou modelar.

| Tarefa | O que fazer |
|--------|-------------|
| **Valores nulos** | Identificar missing (ausente de verdade vs. não coletado), decidir: remover, imputar (média/mediana/moda), ou criar flag “era nulo” |
| **Outliers** | Detectar pontos extremos (IQR, z-score, regras de negócio); tratar com cuidado — às vezes são erros, às vezes são o sinal importante (whales, power users) |
| **Tipos e formatos** | Datas, categorias, IDs — garantir tipos corretos e sem duplicatas inconsistentes |
| **Consistência** | Mesma métrica definida igual em todas as fontes (ex.: “usuário ativo” = DAU ou MAU?) |

> Dados sujos no pré-processamento → EDA enganosa → modelo errado → hipóteses falsas.

---

### 2. Análise exploratória de dados (EDA)

Entender **o que os dados mostram** antes de assumir causas.

- Distribuições (histogramas, box plots)  
- Tendências no tempo (retenção, conversão por cohort)  
- Relações entre variáveis (correlação, segmentos)  
- Comparar grupos (quem converte vs. quem não converte?)  

**Objetivo:** perguntas melhores e suspeitas sobre *onde* investigar — não pular direto para ML.

---

### 3. Feature engineering

Transformar dados brutos em **sinais** que o modelo ou a análise consomem.

Exemplos em produto:

- Dias desde o cadastro, sessões na última semana  
- Taxa de clique em e-mails, features de funil  
- Agregações por usuário / por produto / por campanha  
- Encoding de categorias (país, plano, canal de aquisição)  

Boas features ligam comportamento do usuário ao **KPI** que o negócio quer mover.

---

### 4. Modelagem

Quando EDA e o negócio justificam, entram modelos supervisionados ou não:

- **Classificação:** churn, conversão, abertura de e-mail  
- **Regressão:** CLV, receita prevista, ETA  
- **Não supervisionado:** clusters de usuários para product/marketing  

Comparar baselines, validar, escolher métricas alinhadas ao KPI — não só acurácia técnica.

---

### 5. Geração de hipóteses e insights

O output valioso para product analytics não é só o modelo — são **insights acionáveis** e **hipóteses que dá para testar**.

| Tipo | Exemplo |
|------|---------|
| **Insight** | “Usuários com menos de 3 sessões na primeira semana têm 2× mais churn.” |
| **Hipótese** | “Se enviarmos onboarding personalizado no dia 2, a retenção D30 sobe.” |
| **Experimento** | A/B test, rollout gradual, mudança de feature no app |

O ciclo fecha quando a hipótese vira **experimento mensurável** (métrica + duração + critério de sucesso).

---

### Criar hipóteses que podemos rodar

Boas hipóteses em product analytics são **específicas, falsificáveis e ligadas a dados**:

**Formato útil**

> *Se [intervenção/mudança], então [métrica] vai [subir/cair] em [segmento] em [prazo], porque [razão baseada em EDA/modelo].*

**Checklist antes de rodar**

- [ ] Métrica primária definida (ex.: retenção D7, taxa de conversão)  
- [ ] Segmento claro (novos usuários, mobile, plano free)  
- [ ] Tamanho de amostra / poder estatístico razoável  
- [ ] Duração do teste definida  
- [ ] Pré-processamento e tracking confiáveis para medir o resultado  

**Exemplos de hipóteses testáveis**

1. *Notificação no dia 1 aumenta retorno D3 em usuários que não completaram o tutorial.*  
2. *Recomendações por cluster (descoberto em não supervisionado) aumentam cliques vs. lista genérica.*  
3. *Remover outliers de bots na métrica de sessão não muda a conclusão do teste de pricing.*

---

### Visão integrada (product analytics)

| Etapa | Pergunta que responde |
|-------|------------------------|
| Pré-processamento | Os dados estão limpos e confiáveis? |
| EDA | O que está acontecendo e onde olhar? |
| Feature engineering | Quais sinais descrevem o comportamento? |
| Modelagem | O que conseguimos prever ou segmentar? |
| Hipóteses / insights | O que o time de produto deve **testar** ou **mudar**? |

> **Product analytics** liga dados → entendimento → **hipóteses executáveis** — não para no gráfico nem só no notebook do modelo.

---

## ML para produtos de dados (*data products*)

Um **produto de dados** usa ML para entregar valor contínuo no produto: recomendações, previsões, scores de risco, busca inteligente. O ciclo vai do **enquadramento do problema** até **colocar em produção** — com trade-offs entre **acurácia** e **interpretabilidade**.

```
  Enquadramento → Coleta → Processamento → EDA → Features → Validação cruzada
       → Modelo → Avaliação → Produtização (servir no produto)
```

---

### O que o produto precisa decidir cedo

| Dimensão | Pergunta |
|----------|----------|
| **Acurácia** | Quão errado podemos estar? (SLA de qualidade do modelo) |
| **Interpretabilidade** | O usuário ou o negócio precisa **entender o porquê**? (compliance, confiança, debug) |
| **Tipo de ML** | **Recomendação**, **forecasting** (séries temporais), classificação, regressão… |

Muitas vezes há tensão: modelos mais precisos (deep learning, ensembles grandes) são **menos interpretáveis**; modelos lineares ou com regras explicam melhor, mas podem perder performance.

---

### Tipos comuns em data products

**Recomender (*recommender systems*)**  
- Sugerir próximo vídeo, produto, rota, conteúdo  
- Dados: interações, histórico, perfil, contexto  
- Métricas: clique, tempo assistido, conversão, diversidade  

**Forecasting (previsão)**  
- Prever demanda, tráfego, receita, estoque, ETA  
- Dados: séries temporais, sazonalidade, eventos  
- Métricas: MAE, RMSE, MAPE; erro em horizontes (7d, 30d)  

**Scores e decisões**  
- Churn, fraude, lead score — muitas vezes classificação com threshold no produto  

---

### Pipeline completo (do problema ao ar)

#### 1. Enquadramento do problema (*problem framing*)

- Qual **decisão** o produto toma com o output do modelo?  
- Qual **KPI** melhora (retenção, receita, latência)?  
- Supervisionado ou não? Regressão ou classificação?  
- Baseline sem ML definido antes de investir  

#### 2. Coleta de dados (*data collection*)

- Fontes: eventos do app, warehouse, APIs, logs  
- Granularidade (usuário, sessão, item, dia)  
- Período histórico suficiente; governança e privacidade (LGPD/GDPR)  

#### 3. Processamento de dados (*data processing*)

- Limpeza, nulos, outliers, joins entre tabelas  
- Pipelines reproduzíveis (treino = produção)  
- Feature store ou tabelas agregadas para servir em tempo real  

#### 4. Análise exploratória (EDA)

- Distribuições, sazonalidade, vazamento de dados futuros  
- Validar se há **sinal** antes de modelar caro  

#### 5. Feature engineering

- Sinais alinhados ao momento da predição no produto  
- Features de lag, rolling windows (forecasting), embeddings (recomendação)  

#### 6. Validação cruzada (*cross validation*)

- Evitar overfitting e estimar performance realista  
- **K-fold** para dados i.i.d.; **validação temporal** para forecasting e recomendação (não embaralhar o tempo!)  
- Separar treino / validação / teste com critério documentado  

#### 7. Modelagem

- Treinar candidatos (baseline → modelos mais complexos)  
- Recomendação: collaborative filtering, matrix factorization, two-tower, etc.  
- Forecasting: ARIMA, Prophet, modelos de ML com features de calendário  

#### 8. Avaliação (*evaluation*)

| Contexto | O que medir |
|----------|-------------|
| Offline | Métricas técnicas + proxy de negócio |
| Online | A/B, holdout, impacto no KPI do produto |
| Produto | Latência, disponibilidade, fairness |

Não confundir **boa métrica offline** com **sucesso no produto**.

#### 9. Produtização (*productionalization*)

Colocar o modelo **dentro do data product**:

- API ou batch que gera predições em escala  
- Monitoramento: drift, queda de acurácia, volume  
- Retreino agendado; versionamento de modelos  
- Fallback se o modelo falhar (regra default, cache)  
- Documentação para engenharia, produto e analytics  

---

### Acurácia vs. interpretabilidade (resumo)

| Prioridade | Quando | Exemplos de abordagem |
|------------|--------|------------------------|
| **Interpretabilidade** | Regulação, suporte ao cliente, debug, stakeholders não técnicos | Regressão logística, árvores, SHAP em modelos simples |
| **Acurácia** | Escala, personalização fina, competição por 1% de lift | Ensembles, deep learning, embeddings |
| **Equilíbrio** | Explicar segmentos principais; modelo complexo só onde importa | Modelo campeão + relatório de features importantes |

---

### Checklist — ML em data product

- [ ] Problema e KPI definidos (recomendação **ou** forecasting **ou** score)  
- [ ] Dados coletados e pipeline de processamento estável  
- [ ] EDA e features validadas (sem leakage)  
- [ ] Cross-validation adequada ao tipo de dado  
- [ ] Modelo avaliado offline **e** plano de teste no produto  
- [ ] Produtizado: serve predição, monitora, retreina  

> **Data product** = modelo + dados + engenharia + métrica de negócio — não só o arquivo `.pkl` do modelo.

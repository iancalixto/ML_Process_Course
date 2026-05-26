# Data Collection

Notas do curso **Machine Learning Process** — como **coletar dados** para modelagem e analytics.

> Complementa: [Modeling Process.md](./Modeling%20Process.md) · [Intro to Machine Learning.md](./Intro%20to%20Machine%20Learning.md)

---

## Visão geral das fontes

```
  Bancos SQL (estruturado)     NoSQL / não estruturado     APIs
         │                            │                      │
         └────────────────────────────┴──────────────────────┘
                                  │
                    Web scraping · Pesquisas (surveys)
                                  │
                         Comportamento online · métricas
```

---

## Bancos de dados (SQL)

Na prática, **bancos relacionais** são onde você provavelmente acessa **a maior parte dos dados** do negócio.

**SQL** é ferramenta **vital** para:

- **Extrair** (*pull*) dados de tabelas e views  
- **Juntar** (*join*) entidades (usuário ↔ pedido ↔ produto)  
- **Manipular** agregações, filtros, janelas de tempo  
- Trabalhar com dados **estruturados** em schema definido (linhas, colunas, tipos)

| Vantagem | Uso típico em ML |
|----------|------------------|
| Dados limpos e documentados | Features para treino, cohorts, KPIs históricos |
| Reprodutível | Mesma query = mesmo dataset para retreino |
| Escala | Warehouse (BigQuery, Snowflake, Redshift, PostgreSQL…) |

> Dominar SQL acelera auditoria de dados, EDA e pipelines de feature.

---

## Dados não estruturados e NoSQL

**Bancos não estruturados / NoSQL** guardam formatos flexíveis:

- JSON, documentos, grafos, chave-valor  
- Logs, eventos, feeds, conteúdo de produto  

**Quando aparecem**

- Tipos de dado **específicos** (ex.: posts de rede social, comentários, mídia)  
- Alto volume de eventos com schema que evolui  
- Acesso muitas vezes via **scripts** (SDK, drivers) — **nem sempre** via SQL clássico  

| SQL | NoSQL / scripts |
|-----|-----------------|
| Tabelas relacionais | Documentos, blobs, streams |
| JOINs declarativos | APIs do banco, pipelines ETL |
| Métricas de negócio maduras | Texto, imagem, grafo social |

Combine fontes: features estruturadas no warehouse + sinais não estruturados processados em batch.

---

## APIs (*Application Programming Interfaces*)

APIs permitem que sistemas **compartilhem dados de forma controlada** — você escolhe (dentro do permitido) **o que exatamente pedir**.

**Características**

- Contrato documentado (endpoints, parâmetros, limites de taxa)  
- Autenticação e **permissões** (OAuth, API keys)  
- Respostas em JSON/XML — ideal para integrar em scripts ou pipelines  

**Exemplo: métricas de vídeo (YouTube)**

Ao usar a API de uma plataforma de vídeo, você normalmente define:

| Parâmetro | Exemplo |
|-----------|---------|
| **Métricas** | views, watch time, likes, comentários |
| **Período** | `startDate` / `endDate` |
| **Permissões** | escopo do token (só leitura vs. edição) |
| **Filtros** | canal, vídeo, país, dispositivo |
| **Requisição válida** | parâmetros obrigatórios, formato de data, paginação |

> **Valid request** — requisição que respeita documentação, auth e rate limits; caso contrário: 400/401/403/429.

**Dados de comportamento online**

Muitos produtos expõem eventos via API ou export:

- Cliques, sessões, funil, conversões  
- Integração com analytics (GA4, Mixpanel, Amplitude) ou warehouse próprio  

**Por que APIs importam para ML**

- Dados **atualizados** sem scrape manual  
- Métricas **oficiais** alinhadas ao produto (mesmas definições do negócio)  
- Automação de coleta recorrente para retreino

---

## Comportamento online — o que coletar

**Métricas-chave** dependem do produto, mas costumam incluir:

| Categoria | Exemplos |
|-----------|----------|
| Aquisição | origem, campanha, custo |
| Engajamento | sessões, tempo, páginas, eventos |
| Conversão | signup, compra, trial → paid |
| Retenção | D1, D7, D30, churn |
| Receita | ARPU, LTV, ticket médio |

**Vantagens do online**

- **Fácil de coletar** em escala (pixels, SDKs, server-side events)  
- **Volume massivo** — milhões de eventos por dia  
- **Data store** pode ficar **sempre carregado** (streaming → lake/warehouse) para análise e modelos  

Cuidado: volume alto ≠ qualidade alta — validar definições e **consentimento** (LGPD/cookies).

---

## Web scraping

**Web scraping** = usar **scripts** para baixar dados de uma página ou série de páginas.

**Fluxo comum**

1. Baixar o **HTML** (ou JSON embutido na página)  
2. Parsear com bibliotecas como **Beautiful Soup** (Python)  
3. Extrair campos (título, preço, texto, links)  
4. Salvar em CSV, banco ou pipeline  

**Bot que navega como humano**

- Automação com Selenium / Playwright para **clicar**, rolar, paginar  
- Útil quando o conteúdo só aparece após interação (login, infinite scroll)  
- Mais frágil e lento que API oficial — usar quando não há API

**Ética e legal**

- Ler **termos de uso** e `robots.txt`  
- Respeitar rate limits; não sobrecarregar o site  
- Dados pessoais exigem base legal (LGPD)  

> Preferir **API oficial** quando existir; scraping como último recurso documentado.

---

## Pesquisas (*surveys*)

Quando dados **não existem** no sistema, **criar** dados via survey:

- NPS, satisfação, intenção de compra  
- Labels para treino supervisionado (rotular amostra manualmente)  
- Complementar gaps de tracking no app  

| Prós | Contras |
|------|---------|
| Perguntas sob medida | Viés de resposta, amostra pequena |
| Barato para pilotos | Escala limitada vs. logs automáticos |

Combine surveys (qualitativo + labels) com dados comportamentais (quantitativo em massa).

---

## Comparativo rápido

| Fonte | Formato | Ferramenta típica | Melhor para |
|-------|---------|-------------------|-------------|
| SQL / warehouse | Estruturado | SQL | KPIs, features, histórico |
| NoSQL | Semi/não estruturado | Scripts, drivers | Eventos, JSON, social |
| APIs | JSON contratado | HTTP client, SDK | Métricas oficiais, integrações |
| Comportamento online | Eventos | Tracking + warehouse | ML em escala, produto digital |
| Web scraping | HTML → tabelas | Beautiful Soup, bots | Dados públicos sem API |
| Surveys | Respostas | Forms, Typeform | Labels, opinião, hipóteses |

---

## Checklist — coleta de dados

- [ ] Fontes mapeadas (SQL, NoSQL, API, scrape, survey)  
- [ ] Permissões e termos legais OK  
- [ ] Período e granularidade definidos  
- [ ] Métricas-chave alinhadas ao problema de negócio  
- [ ] Requisições de API válidas e reproduzíveis  
- [ ] Pipeline documentado (como repetir a coleta)  

---

## Suas notas

- 
- 

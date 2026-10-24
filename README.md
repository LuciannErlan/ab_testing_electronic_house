# Teste A/B — Electronic House: Preenchimento Manual vs. Automático no Checkout

## 1. A Empresa Electronic House (Problema de Negócio)
A **Electronic House** é um comércio eletrônico (*e-commerce*) de produtos de informática para casas e escritórios. Os clientes podem adquirir mouses, monitores, teclados, computadores, laptops, cabos HDMI, fones de ouvido, webcams, entre outros itens, recebendo os produtos no conforto de suas residências.

Com atuação global — presente não apenas no Brasil, mas também em diversos países da América do Norte, América Latina e Europa —, o Diretor de Produtos Global solicitou ao Head de Design uma nova forma de finalizar compras com cartão de crédito, eliminando a necessidade de digitação manual de todos os campos e garantindo funcionamento consistente em todos os países.

Após meses de desenvolvimento, o time de Engenharia Backend entregou uma solução de pagamento na qual **90% das informações do formulário são preenchidas automaticamente**. Para validar sua efetividade antes de uma adoção definitiva, as duas versões da página de vendas foram colocadas no ar simultaneamente, e o time de Front-End implementou uma automação que rotulou cada cliente conforme a versão visualizada:
- **`GRP A` (Grupo Tratamento):** Preenchimento **Automático** (90% dos dados preenchidos automaticamente).
- **`GRP B` (Grupo Controle):** Preenchimento **Manual** do formulário de cartão de crédito.

---

## 2. O Desafio
Avaliar com rigor estatístico os dados coletados durante o experimento (`ab_testing.csv`) para apoiar o time de Design e o Diretor de Produtos Global na tomada de decisão, respondendo à pergunta principal:

> **Qual a melhor forma de pagamento: Preenchimento Manual (`GRP B`) ou Automático (`GRP A`) do formulário de dados do cartão de crédito?**

---

## 3. Dicionário de Dados (`ab_testing.csv`)

| Variável | Descrição |
| :--- | :--- |
| `uid` | Identificador único do cliente na plataforma. |
| `country` | Código ISO de 3 letras do país do cliente (`BRA`, `USA`, `MEX`, `DEU`, `ESP`, `FRA`, `GBR`, `TUR`, `AUS`, `CAN`). |
| `gender` | Gênero informado pelo cliente (`F`, `M`, entre outros). |
| `spent` | Valor monetário total gasto pelo cliente durante o período avaliado. |
| `purchases` | Quantidade total de compras realizadas pelo cliente. |
| `date` | Data de registro (`YYYY-MM-DD`). |
| `group` | Grupo experimental atribuído ao cliente (`GRP A` = Preenchimento Automático; `GRP B` = Preenchimento Manual). |
| `device` | Canal de compra utilizado (`I` = Compra feita através do **Site**; `A` = Compra feita através do **App**). |

---

## 4. Estratégia de Solução
A análise foi desenvolvida no notebook [`ab_testing_electronic_house.ipynb`](./ab_testing_electronic_house.ipynb) seguindo cinco etapas:

1. **Auditoria de Qualidade dos Dados:** verificação de valores ausentes, duplicidade de `uid`, ausência de contaminação de usuários entre grupos e consistência entre `spent` e `purchases`.
2. **Validação do Desenho Experimental:** cálculo do tamanho mínimo amostral (*Power Analysis* com $\alpha = 5\%$ e poder $1 - \beta = 80\%$), teste Qui-quadrado para *Sample Ratio Mismatch* (divisão 50/50) e homogeneidade demográfica/canal entre grupos.
3. **Análise Exploratória de Dados (EDA):** inspeção das distribuições de gasto por cliente (`spent`), frequência de compras (`purchases`), ticket médio (`avg_ticket`) e evolução temporal mensal da receita.
4. **Inferência Estatística e Testes de Hipóteses:** verificação de normalidade (D'Agostino-Pearson e Shapiro-Wilk) e homocedasticidade (Levene), seguida pelo **Teste *t* de Welch**, **Teste *t* de Student**, **Teste Não-Paramétrico de Mann-Whitney U** e **Intervalo de Confiança de 95% via *Bootstrapping*** (2.000 reamostragens) para o *lift* do Automático (`GRP A`) sobre o Manual (`GRP B`).
5. **Análise Segmentada Global:** avaliação do desempenho e significância estatística em cada país (com correção de **Bonferroni** para múltiplas comparações), canal de compra (`I = Site` vs. `A = App`) e gênero (`gender`).

---

## 5. Estrutura do Repositório

```text
ab_testing_electronic_house/
├── .gitignore                         # Arquivos temporários e caches ignorados pelo Git
├── ab_testing.csv                     # Base de dados do experimento A/B
├── ab_testing_electronic_house.ipynb  # Notebook com EDA, testes estatísticos e conclusões
└── README.md                          # Documentação executiva do projeto
```

---

## 6. Como Executar
1. Abra o arquivo `ab_testing_electronic_house.ipynb` no VS Code / Antigravity ou Jupyter Lab.
2. Selecione um kernel Python 3 com as bibliotecas `pandas`, `numpy`, `scipy`, `matplotlib` e `seaborn`.
3. Clique em **Run All (Executar Tudo)** para gerar todas as tabelas estatísticas, gráficos e diagnósticos.

---

## 7. Resultados e Conclusão Executiva

A execução completa sobre os **45.883 registros** (`23.009` em `GRP A` Automático e `22.874` em `GRP B` Manual) demonstrou **ausência de diferença estatisticamente significativa** entre os dois métodos de pagamento ($\alpha = 0{,}05$):

| Métrica | GRP B (Manual) | GRP A (Automático) | Diferença (A − B) | Lift (%) | p-valor (Welch *t*) | p-valor (Mann-Whitney) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gasto Médio (`spent`)** | 1.886,42 | 1.880,50 | -5,92 | -0,31% | 0,6423 | 0,4500 |
| **Compras por Cliente (`purchases`)** | 4,5525 | 4,5364 | -0,0161 | -0,35% | 0,5782 | 0,3594 |
| **Ticket Médio (`avg_ticket`)** | 415,03 | 415,07 | +0,04 | +0,01% | 0,9726 | 0,9752 |

- **Intervalo de Confiança de 95% (*Bootstrapping*, 2.000 reamostragens):** `[-30,87; +19,14]` para a diferença de gasto médio (`[-1,62%; +1,02%]` em *lift*), cruzando o zero.
- **Canais (`Site` vs. `App`):** sem diferença estatisticamente significativa tanto no **Site (`I`)** (*p* = `0,8733`) quanto no **App (`A`)** (*p* = `0,6202`).
- **Recorte por País:** nenhum dos 10 países apresentou ganho estatisticamente significativo com o preenchimento automático; além disso, **Espanha (`ESP`)** registrou retração de **-8,21%** (`1.794,19` vs. `1.954,68`, Welch *p* = `0,0068`, Mann-Whitney *p* = `0,0039`), sugerindo possível atrito técnico regional no autopreenchimento.

**Recomendação Final:** Na configuração atual, o **Preenchimento Automático (`GRP A`) não se mostrou superior ao Preenchimento Manual (`GRP B`)**. Recomenda-se não substituir globalmente o checkout manual antes de auditar o comportamento técnico do autopreenchimento em mercados como Espanha e Alemanha e incluir logs de abandono de checkout (`purchases = 0`) em um novo ciclo experimental.

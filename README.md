# stocks-prediction

Uma pipeline de pesquisa em Python para previsão de retornos de ações e backtesting de estratégias, combinando PCA, Redes Neurais (MLP) e validação walk-forward com re-treino semanal.

O projeto cobre o fluxo completo: aquisição de dados, engenharia de features, treino dos modelos e simulação de trades com custos de transação.

---

## Sumário

- [Visão geral](#visão-geral)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
  - [1. Com vs. sem regressor](#1-com-vs-sem-regressor)
  - [2. Backtest com vs. sem re-treino semanal](#2-backtest-com-vs-sem-re-treino-semanal)
- [Limitações](#limitações)

---

## Visão geral

O objetivo é prever o movimento de preços de ações no dia seguinte e avaliar a estratégia resultante em um backtest realista, sem **look-ahead bias**.

Componentes principais:

- **PCA** sobre os retornos do mercado, para extrair fatores comuns entre ativos.
- **MLPClassifier** — classifica se o trade (com custos) vale a pena ser aberto.
- **MLPRegressor** (opcional) — prevê o retorno contínuo e é usado como filtro de entrada no backtest.
- **Walk-forward** — a cada semana, os modelos são re-treinados apenas com dados do passado recente (janela deslizante) e aplicados na semana seguinte.
- **Backtest** — simulador de trades com `buy_sell_fee` e `commission_bps`, calculando PnL acumulado e comparação com o benchmark (buy & hold).

---

## Instalação

O projeto usa [uv](https://github.com/astral-sh/uv) para gerenciar o ambiente e as dependências.


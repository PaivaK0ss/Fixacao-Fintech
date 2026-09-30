import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =-=-=-=-=-=-=-=
#   Atividade
# =-=-=-=-=-=-=-=

# Carregamento do arquivo
df = pd.read_csv(
    'transacoes.csv',
    encoding='latin1'
)

# Correção de dados faltantes usando a mediana dos valores
mediana_por_estado = (df.groupby('estado_cliente')['valor'].median())
df['valor'] = df['valor'].fillna(mediana_por_estado)

# Coluna estática para rastreamento do pipeline
df['plataforma'] = 'Mobile'

# Conversão da data para DateTime e localização para America/Sao_Paulo
df['data_transacao'] = pd.to_datetime(df['data_transacao'])

df['data_transacao'] = (
    df['data_transacao']
      .dt
      .tz_localize('America/Sao_Paulo')
)


# Extração do dia da semana e do mês
df['dia_semana'] = (
    df['data_transacao']
      .dt
      .day_name()
)

df['mes'] = (
    df['data_transacao']
      .dt
      .month
)

# Remoção de duplicatas
df = df.drop_duplicates(
    keep='first'
)

# Filtro de transações de setembro que atendem aos requisitos: realizadas no estado 'SP' ou 'RJ' e possuem valor da transação maior que R$ 5.000,00
filtro_transacao = (
    (df['mes'] == 9)
    &
    (
        (df['estado_cliente'] == 'SP')
        |
        (df['estado_cliente'] == 'RJ')
    )
    &
    (df['valor'] > 5000)
)

df_filtrado = df[filtro_transacao].copy() # Aplicação do filtro a um DataFrame exclusivo

# Dicionário contendo a categoria de risco de cada cliente
risco_dict = {
    'C100': 'Baixo',
    'C101': 'Alto',
    'C102': 'Médio',
    'C103': 'Baixo',
    'C104': 'Alto'
}

# Mapeamento do DataFrame com base no dicionário
df['nivel_risco'] = (
    df['id_cliente']
    .map(risco_dict)
)

# Criação da pivot_table com base no df nível_risco
tabela_risco = pd.pivot_table(
    df,
    index='mes',
    columns='nivel_risco',
    values='valor',
    aggfunc='sum',
    margins=True
)

# Média das transações por estado
media_estado = (
    df.groupby('estado_cliente')['valor']
    .transform('mean')
)

# Calcula o desvio padrão das transações de cada estado.
desvio_estado = (
    df.groupby('estado_cliente')['valor']
    .transform('std')
)

# Cálculo do Z-Score
df['z_score'] = (
    (df['valor'] - media_estado) / desvio_estado
)

# Análise de anomalias
df_anomalias = df[df['z_score'] > 2.5].copy()

# Preparação dos dados para o gráfico
df['data_dia'] = (
    df['data_transacao']
    .dt
    .floor('D') # arredonda cada data para o início daquele dia
)

# Agrupando as transações pela data.
totais_diarios = (
    df.groupby('data_dia')['valor']
    .sum()
    .sort_index()
)

# Média móvel de 7 dias
media_movel = (
    totais_diarios
    .rolling(7)
    .mean()
)

# Visualização com MatPlotLib

# Criação da figura
fig, ax = plt.subplots(
    figsize=(12, 6)
)

# Desenha a linha contendo o valor total das transações de cada dia
ax.plot(
    totais_diarios.index,
    totais_diarios.values,
    label='Total diário'
)

# Desenha a média móvel de 7 dias.
ax.plot(
    media_movel.index,
    media_movel.values,
    label='Média móvel de 7 dias'
)

# Faz o eixo Y começar em zero.
ax.set_ylim(
    bottom=0
)
# Define o título.
ax.set_title(
    'Valor Total de Transações e Média Móvel de 7 Dias'
)
# Nome do eixo X.
ax.set_xlabel(
    'Data'
)
# Nome do eixo Y.
ax.set_ylabel(
    'Valor das Transações (R$)'
)
# Exibe a legenda das duas linhas.
ax.legend()

# Gira as datas do eixo X para facilitar a leitura.
plt.xticks(
    rotation=45
)

# Ajusta automaticamente os elementos do gráfico para evitar sobreposição.
plt.tight_layout()

# Exibe o gráfico.
plt.show()
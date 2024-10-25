# Importação de bibliotecas

# Manipulação e tratamento de dados
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
file_path= "base/MiningProcess_Flotation_Plant_Database.csv"

dados = pd.read_csv(
    file_path,
    decimal=',',
)

# Pré-processamento

print(dados.head()) # Exibir as primeiras linhas do dataset
print(dados.info()) # Obteção de informações do dataset
print(dados.describe()) # Verificar a distribuição dos valores de cada coluna

# A coluna data está formatada como objet. É necessário transformá-la em um tipo data
# Não existe números nulos e nem números negativos, indicando que não há valores fora do esperado

dados['date'] = pd.to_datetime(dados['date'], format='%Y-%m-%d %H:%M:%S') # Conversão da coluna 'date' para datetime
dados.set_index('date', inplace=True) # Definir a coluna 'date' como índice do DataFrame original
print(dados.info()) # Verificar a mudança de tipo da coluna 'date'

# Verificar linhas duplicadas
linhas_duplicadas = dados.duplicated().sum()
#print(dados[dados.duplicated()]) # Exibir as linhas duplicadas
dados_iniciais = dados.shape[0] # Número de linhas do dataset inicial
porcentagem_duplicadas = (linhas_duplicadas / dados_iniciais) * 100
print("Existem {} linhas duplicadas, o que equivale a {:.2f}% do dataset inicial.".format(linhas_duplicadas, porcentagem_duplicadas))

# Verificar se existem dias faltosos no intervalo de tempo
dados_faltosos = dados[~dados.index.duplicated(keep='first')] # Remover duplicatas do índice
date_range = pd.date_range(start=dados_faltosos.index.min(), end=dados_faltosos.index.max(), freq='D') # Criar um índice de datas que cobre todo o intervalo de tempo do dataset
dados_reindexed = dados_faltosos.reindex(date_range) # Reindexar o DataFrame para incluir todas as datas do intervalo
dias_faltando = dados_reindexed[dados_reindexed.isnull().all(axis=1)].index # Identificar os dias que estão faltando
print('Dias faltando no intervalo de tempo:')
print(dias_faltando)

# Existem dias faltando no intervalo de tempo, o que pode indicar que o dataset não está completo
# Uma alternativa consiste em preencher os valores faltantes com a média dos valores anteriores e posteriores

# Verificar consistência das medidas
# Para cada hora devem haver 180 medidas, uma vez que cada medida é feita a cada 20 segundos
contagem_medidas = dados.groupby(dados.index).count()
print(contagem_medidas[contagem_medidas['% Iron Feed'] != 180])

# Existem dias que não possuem 180 medidas, no entanto, são apenas dois dias o que não causa impacto signifiticativo

# Caso se opte em não preencher os valores faltantes, uma alterativa é selecionar os dados a partir de uma data específica
# Por exemplo, a partir de 2017-03-29, para retirar os dias faltantes e reduzir o impacto nas análises

dados_tratados= dados.groupby(dados.index).mean() # Agrupar os dados pela média diária
#dados_tratados = dados['date'] > '2017-03-29' # Selecionar os dados a partir de 2017-03-29
#print(dados_tratados.describe())
#print(dados_tratados.info())

dados_tratados.to_csv('base/dados_tratados_media.csv', index=True) # Salvar os dados tratados em um novo arquivo CSV
dados.to_csv('base/dados_tratados.csv', index=True) # Salvar os dados tratados em um novo arquivo CSV

# EDA - Análise Exploratória de Dados

# HISTOGRAMA SILICA

fig, ax = plt.subplots(figsize=(8, 4))
n, bins, patches = ax.hist(dados_tratados['% Silica Concentrate'], bins=50, edgecolor='black', alpha=0.7, label='% Iron Feed')
# Definir cores personalizadas para as barras
for i in range(len(patches)):
    if bins[i] < 3:
        patches[i].set_facecolor('#011638')
    else:
        patches[i].set_facecolor('#c6c7c4')

#ax.legend(loc='upper right')
ax.set_title("Concentração de sílica após o beneficiamento", loc='left', fontsize=16, pad=20, color = '#353b3c')
ax.set_ylabel("Número de ocorrências")
ax.set_xlabel("% de concentração de silica")
ax.set_facecolor('white')  # Definir fundo branco

# Incluir texto no gráfico
ax.text(x=3.5,y=200, s="$\\bf{Após}$ o beneficiamento \n"
        "a $\\bf{concentração}$ de silica\n"
        "é reduzida drasticamente,\n"
        "se mantendo em grande parte \n" 
        "$\\bf{menor}$ que 3%.",
        verticalalignment='top',
        fontsize=14, color='#353b3c')
ax.spines['top'].set_visible(False)  # Remover a linha superior
ax.spines['right'].set_visible(False)  # Remover a linha direita

plt.savefig('Images/Histo_silica.png')


# SERIE TEMPORAL SILICA
cor_dados = ['#c6c7c4', '#011638']
fig, ax = plt.subplots(figsize=(16, 9))  
variaveis = ["% Silica Feed", "% Silica Concentrate"]
novos_nomes = ["Concentrado inicial", "Concentrado final"]

for i, var in enumerate(variaveis):
    ax.plot(dados.index, dados[var], label=novos_nomes[i], color=cor_dados[i])
    ax.fill_between(dados.index, dados[var], color=cor_dados[i], alpha=1)
    

ax.legend(loc='upper right')
ax.set_title("Concentração de sílica $\\bf{ANTES}$ e $\\bf{APÓS}$  o beneficiamento", loc='Left', fontsize=22, pad=40, color = '#353b3c')    

leg = ax.legend(loc='upper right', fontsize=16, frameon=True, framealpha=0.9, facecolor='white', edgecolor='white')
for line in leg.get_lines():
    line.set_linewidth(18)  # Aumentar a espessura das linhas na legenda


#ax.set_title("Histograma de {}".format(var))
ax.set_ylabel("% Concentração de Silica", fontsize=16)
#ax.set_xlabel(" ", fontsize=16)
ax.set_facecolor('white')  # Definir fundo branco
ax.spines['top'].set_visible(False)  # Remover a linha superior
ax.spines['right'].set_visible(False)  # Remover a linha direita
# Incluir texto no gráfico
ax.text(x=pd.Timestamp('2017-03-10'),y=35, s="$\\bf{Após}$ o beneficiamento"
        "a $\\bf{concentração}$ de sílica \n"
        "sofre uma $\\bf{redução}$ considerável.\n",
        verticalalignment='top',
        fontsize=20, color='#353b3c')
# Ajustar o tamanho da fonte dos ticks dos eixos x e y
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# Ajustar layout e mostrar o gráfico
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar layout para não sobrepor o título
plt.savefig('Images/serie_temporal_silica.png')

# HISTOGRAMA FERRO

fig, ax = plt.subplots(figsize=(8, 4))
n, bins, patches = ax.hist(dados_tratados['% Iron Concentrate'], bins=50, edgecolor='black', alpha=0.7, label='% Iron Feed')
# Definir cores personalizadas para as barras
for i in range(len(patches)):
    if bins[i] > 64:
        patches[i].set_facecolor('#011638')
    else:
        patches[i].set_facecolor('#c6c7c4')

#ax.legend(loc='upper right')
ax.set_title("Concentração de ferro após o beneficiamento", loc='left', fontsize=16, pad=20, color = '#353b3c')
ax.set_ylabel("Número de ocorrências")
ax.set_xlabel("% de concentração de ferro")
ax.set_facecolor('white')  # Definir fundo branco

# Incluir texto no gráfico
ax.text(x=66,y=250, s="O concentrado de $\\bf{ferro}$ \n"
        "fica em grande parte \n"
        "$\\bf{maior}$ que 64%.",
        verticalalignment='top',
        fontsize=14, color='#353b3c')
ax.spines['top'].set_visible(False)  # Remover a linha superior
ax.spines['right'].set_visible(False)  # Remover a linha direita

plt.savefig('Images/Histo_ferro.png')

# SERIE TEMPORAL FERRO

cor_dados = ['#c6c7c4', '#011638']
fig, ax = plt.subplots(figsize=(16, 9))  
variaveis = ["% Iron Feed", "% Iron Concentrate"]
novos_nomes = ["Concentrado inicial", "Concentrado final"]

for i, var in enumerate(variaveis):
    ax.plot(dados.index, dados[var], label=novos_nomes[i], color=cor_dados[i])
    ax.fill_between(dados.index, dados["% Iron Concentrate"],  dados["% Iron Feed"], color='#011638', alpha=1)
    ax.fill_between(dados.index, dados["% Iron Feed"], color='#c6c7c4', alpha=0.4)


ax.legend(loc='upper right')
ax.set_title("Concentração de ferro $\\bf{ANTES}$ e $\\bf{APÓS}$  o beneficiamento", loc='Left', fontsize=22, pad=40, color = '#353b3c')    

leg = ax.legend(loc='upper right', fontsize=16, frameon=True, framealpha=0.9, facecolor='white', edgecolor='white')
for line in leg.get_lines():
    line.set_linewidth(18)  # Aumentar a espessura das linhas na legenda

ax.set_ylim(40, 75)
ax.set_ylabel("% Concentração de ferro", fontsize=16)
#ax.set_xlabel(" ", fontsize=16)
ax.set_facecolor('white')  # Definir fundo branco
ax.spines['top'].set_visible(False)  # Remover a linha superior
ax.spines['right'].set_visible(False)  # Remover a linha direita
# Incluir texto no gráfico
ax.text(x=pd.Timestamp('2017-03-10'),y=75, s="O beneficiamento produz ferro em concentração \n$\\bf{maior}$"
        "que $\\bf{60\\%}$",
        verticalalignment='top',
        fontsize=20, color='#353b3c')
# Ajustar o tamanho da fonte dos ticks dos eixos x e y
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# Ajustar layout e mostrar o gráfico
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar layout para não sobrepor o título
plt.savefig('Images/serie_temporal_ferro.png')





# # Analise dos dados de concentração de sílica e ferro
cor_dados = ['#6c757d', '#adb5bd', '#7f5539', '#b08968']
fig, axs = plt.subplots(2, 2, figsize=(16, 18))  # Criar uma figura com 4 subplots
#fig.suptitle('Séries temporais das variáveis de interesse', fontsize=20)  # Adicionar um título à figura
variaveis = ['% Iron Feed', '% Iron Concentrate', "% Silica Feed", "% Silica Concentrate"]
for i, var in enumerate(variaveis):
    ax = axs[i // 2, i % 2]  # Acessar o subplot correto
    ax.hist(dados[var], bins=50, color=cor_dados[i], edgecolor='black', alpha=0.7, label=var)
    ax.legend(loc='upper right')
    ax.set_title("Histograma de {}".format(var))
    ax.set_ylabel("Número de ocorrências")
    ax.set_xlabel(var)
    ax.set_facecolor('white')  # Definir fundo branco

# Ajustar layout e mostrar o gráfico
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar layout para não sobrepor o título
plt.savefig('Images/iron_feed_concentrade.png')



# Análise de correlação
matriz_correlacao = dados_tratados.corr()

mask = np.triu(np.ones_like(matriz_correlacao, dtype=bool))# Criar uma máscara para o triângulo superior
plt.figure(figsize=(20, 15))
sns.set(font_scale=1.0)
heatmap = sns.heatmap(matriz_correlacao, annot=True, cmap='RdYlBu', square=True,
                      fmt=".2f",  # Exibir valores com duas casas decimais
                      center=0,
                      xticklabels=matriz_correlacao.columns,  # Rótulos do eixo x
                      yticklabels=matriz_correlacao.columns,  # Rótulos do eixo y
                      annot_kws={"size": 15},  # Tamanho da fonte dentro das células
                      linewidths=0.5, linecolor='white',  # Bordas das células
                      mask=mask,
                      vmin=-1, vmax=1,  # Escala de cores de -1 a 1
                      cbar_kws={"shrink": 1})  # Ajustar o tamanho da barra de cores


ax = heatmap.axes# Obter os eixos atuais
ax.set_aspect("auto")# Ajustar o aspecto dos eixos para 'auto' para ajustar automaticamente o tamanho das células

plt.title("Mapa de correlação entre as variáveis")# Adicionar um título ao heatmap
plt.savefig('Images/mapa_de_correlacao.png')

# Separação das features do dataset

airFlow = dados_tratados[['Flotation Column 01 Air Flow', 
                          'Flotation Column 02 Air Flow', 
                          'Flotation Column 03 Air Flow', 
                          'Flotation Column 04 Air Flow',
                          'Flotation Column 05 Air Flow', 
                          'Flotation Column 06 Air Flow', 
                          'Flotation Column 07 Air Flow',
                          '% Silica Concentrate'
                        ]].drop_duplicates().dropna().groupby('date').mean()  

airLevel = dados_tratados[['Flotation Column 01 Level',
                          'Flotation Column 02 Level', 
                          'Flotation Column 03 Level', 
                          'Flotation Column 04 Level', 
                          'Flotation Column 05 Level', 
                          'Flotation Column 06 Level', 
                          'Flotation Column 07 Level',
                          '% Silica Concentrate'
                          ]].drop_duplicates().dropna().groupby('date').mean()

importante = dados_tratados[['Starch Flow',
                            'Amina Flow',
                            'Ore Pulp Flow',
                            'Ore Pulp pH',
                            'Ore Pulp Density',
                            '% Silica Concentrate',
                            ]].drop_duplicates().dropna().groupby('date').mean()

# Calcular a correlação
correlacao = importante.corr()
mask = np.triu(np.ones_like(correlacao, dtype=bool))  # Criar uma máscara para o triângulo superior

# Criar o mapa de calor
plt.figure(figsize=(20, 15))
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlacao, annot=True, cmap='RdYlBu', square=True,
                      fmt=".2f", center=0, xticklabels=correlacao.columns,
                      yticklabels=correlacao.columns, annot_kws={"size": 15},
                      linewidths=0.5, linecolor='white', mask=mask, vmin=-1, vmax=1,
                      cbar_kws={"shrink": 1})
heatmap.axes.set_aspect("auto")
plt.title("Mapa de correlação entre as variáveis")
plt.savefig('Images/mapa_de_correlacao_importante.png')

# Visualização das séries temporais
cor_importante = ['#9dc6ae', '#bbd0ff', '#c8b6ff', '#deab90']
fig, axs = plt.subplots(4, 1, figsize=(16, 18))  # Criar uma figura com 4 subplots
fig.suptitle('Séries temporais das variáveis de interesse', fontsize=20)  # Adicionar um título à figura

# Adicionar as séries temporais aos subplots
variaveis = ['Starch Flow', 'Amina Flow', 'Ore Pulp pH', '% Silica Concentrate']
for i, var in enumerate(variaveis):
    axs[i].plot(importante.index, importante[var], label=var, color=cor_importante[i])
    axs[i].set_ylabel(var)
    axs[i].legend()
    axs[i].set_facecolor('white')  # Definir fundo branco
    decomposicao = seasonal_decompose(importante[var], model='additive', period=30, extrapolate_trend=30)
    axs[i].plot(decomposicao.trend, color='red')

# Adicionar rótulo ao eixo x no último subplot
axs[-1].set_xlabel('Date')

# Ajustar layout e mostrar o gráfico
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar layout para não sobrepor o título
plt.savefig('Images/series_temporais_importante.png')

# Calcular a correlação
correlacao_airLevel = airLevel.corr()
mask = np.triu(np.ones_like(correlacao_airLevel, dtype=bool))  # Criar uma máscara para o triângulo superior

# Criar o mapa de calor
plt.figure(figsize=(20, 15))
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlacao_airLevel, annot=True, cmap='RdYlBu', square=True,
                      fmt=".2f",  # Exibir valores com duas casas decimais
                      center=0,
                      xticklabels=correlacao_airLevel.columns,  # Rótulos do eixo x
                      yticklabels=correlacao_airLevel.columns,  # Rótulos do eixo y
                      annot_kws={"size": 15},  # Tamanho da fonte dentro das células
                      linewidths=0.5, linecolor='white',  # Bordas das células
                      mask=mask,
                      vmin=-1, vmax=1,  # Escala de cores de -1 a 1
                      cbar_kws={"shrink": 1})  # Ajustar o tamanho da barra de cores

# Ajustar o aspecto dos eixos para 'auto' para ajustar automaticamente o tamanho das células
heatmap.axes.set_aspect("auto")

# Adicionar um título ao heatmap
plt.title("Mapa de correlação entre as variáveis de Air Level")
plt.savefig('Images/mapa_de_correlacao_airLevel.png')

# Visualização das séries temporais
# DATAFRAME: airLevel
cor_airLevel = ['#e4e7e4', '#c0c4ca', '#9ba1b0', '#777f96', '#535c7b', '#2e3961', '#0a1647','#deab90']
fig, axs = plt.subplots(8, 1, figsize=(16, 24))  # Criar uma figura com 7 subplots
fig.suptitle('Séries temporais das variáveis de Air Level', fontsize=20)  # Adicionar um título à figura

# Adicionar as séries temporais aos subplots
for i, coluna in enumerate(airLevel.columns):
    axs[i].plot(airLevel.index, airLevel[coluna], label=coluna, color=cor_airLevel[i])
    axs[i].set_ylabel(coluna)
    axs[i].legend()
    axs[i].set_facecolor('white')  # Definir fundo branco
    decomposicao = seasonal_decompose(airLevel[coluna], model='additive', period=30, extrapolate_trend=30)
    axs[i].plot(decomposicao.trend, color='red')

# Adicionar rótulo ao eixo x no último subplot
axs[-1].set_xlabel('Date')

# Ajustar layout e mostrar o gráfico
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar layout para não sobrepor o título
plt.savefig('Images/series_temporais_airLevel.png')


# Calcular a correlação
correlacao_airFlow = airFlow.corr()
mask = np.triu(np.ones_like(correlacao_airFlow, dtype=bool))  # Criar uma máscara para o triângulo superior

# Criar o mapa de calor
plt.figure(figsize=(20, 15))
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlacao_airFlow, annot=True, cmap='RdYlBu', square=True,
                      fmt=".2f",  # Exibir valores com duas casas decimais
                      center=0,
                      xticklabels=correlacao_airFlow.columns,  # Rótulos do eixo x
                      yticklabels=correlacao_airFlow.columns,  # Rótulos do eixo y
                      annot_kws={"size": 15},  # Tamanho da fonte dentro das células
                      linewidths=0.5, linecolor='white',  # Bordas das células
                      mask=mask,
                      vmin=-1, vmax=1,  # Escala de cores de -1 a 1
                      cbar_kws={"shrink": 1})  # Ajustar o tamanho da barra de cores

# Ajustar o aspecto dos eixos para 'auto' para ajustar automaticamente o tamanho das células
heatmap.axes.set_aspect("auto")

# Adicionar um título ao heatmap
plt.title("Mapa de correlação entre as variáveis de Air Flow")
plt.savefig('Images/mapa_de_correlacao_airFlow.png')

# Visualização das séries temporais
# DATAFRAME: airFlow
cor_airFlow = ['#edf2fb', '#e2eafc', '#d7e3fc', '#ccdbfd', '#c1d3fe', '#b6ccfe', '#abc4ff','#deab90']
fig, axs = plt.subplots(8, 1, figsize=(16, 24))  # Criar uma figura com 7 subplots
fig.suptitle('Séries temporais das variáveis de Air Flow', fontsize=20)  # Adicionar um título à figura

# Adicionar as séries temporais aos subplots
for i, coluna in enumerate(airFlow.columns):
    axs[i].plot(airFlow.index, airFlow[coluna], label=coluna, color=cor_airFlow[i])
    axs[i].set_ylabel(coluna)
    axs[i].legend()
    axs[i].set_facecolor('white')  # Definir fundo branco
    decomposicao = seasonal_decompose(airFlow[coluna], model='additive', period=30, extrapolate_trend=30)
    axs[i].plot(decomposicao.trend, color='red')

# Adicionar rótulo ao eixo x no último subplot
axs[-1].set_xlabel('Date')

# Ajustar layout e mostrar o gráfico
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar layout para não sobrepor o título
plt.savefig('Images/series_temporais_airFlow.png')

# Uma analise visual rápida das series temporais indicam que, de certa maneira, 
# O pH baixo da polpa de minério, a redução de amido e amida está associado a um aumento na concentração de sílica
# Controlar esses parâmetros pode ser uma maneira de controlar a concentração de sílica

# O controle do níveis de fluxo ar nas colunas de flotação finais pode contribuir para a redução da silica

# APlicação da técnica PCA para redução de dimensionalidade

x = dados_tratados
scaler = StandardScaler() # Padronização dos dados, os dados possuem escalas diferentes a padronização é necessária
x_scaled = scaler.fit_transform(x)

# Aplicação da PCA sem especificar o número de componentes
pca = PCA()
pca.fit(x_scaled)

# Variância explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.savefig('Images/scree_plot.png')


# Cumulative explained variance plot
cumulative_explained_variance = explained_variance.cumsum()
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.axhline(y=0.8, color='r', linestyle='--')  # Linha de referência para 85% da variância explicada
plt.savefig('Images/cumulative_explained_variance.png')


# Determinar o número ótimo de componentes principais
optimal_components = next(i for i, cumulative_variance in enumerate(cumulative_explained_variance) if cumulative_variance >= 0.8) + 1
print(f'O número ótimo de componentes principais é: {optimal_components}')

# Aplicação da PCA com o número ótimo de componentes
pca_optimal = PCA(n_components=optimal_components)
principal_components = pca_optimal.fit_transform(x_scaled)

# Criação de um DataFrame com os componentes principais
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(optimal_components)])
pca_df.index = dados_tratados.index  # Manter o índice de tempo original


# Análise das cargas dos componentes principais
loadings = pd.DataFrame(pca_optimal.components_.T, columns=[f'PC{i+1}' for i in range(optimal_components)], index=dados_tratados.columns)
print(loadings)   

# Visualização das cargas dos componentes principais
plt.figure(figsize=(16, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm')
plt.title('Heatmap of Principal Component Loadings')
plt.xlabel('Principal Components')
plt.ylabel('Features')
plt.savefig('Images/pca_loadings_heatmap.png')


# Visualização das séries temporais dos componentes principais
plt.figure(figsize=(16, 8))
for i in range(optimal_components):
    plt.plot(pca_df.index, pca_df[f'PC{i+1}'], label=f'PC{i+1}')
plt.xlabel('Date')
plt.ylabel('Principal Components')
plt.legend()
plt.title('Principal Components over Time')
plt.savefig('Images/pca_time_series.png')

# Importação de bibliotecas

# Manipulação e tratamento de dados
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


# Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
file_path= "base/MiningProcess_Flotation_Plant_Database.csv"

dados = pd.read_csv(
    file_path,
    decimal=',',
)
dados['date'] = pd.to_datetime(dados['date'], format='%Y-%m-%d %H:%M:%S') # Conversão da coluna 'date' para datetime
dados.set_index('date', inplace=True) # Definir a coluna 'date' como índice do DataFrame original
#dados = dados.astype('float64')

dados_tratados= dados.groupby(dados.index).mean().dropna() # Agrupar os dados pela média diária

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



# Visualização das series temporais 

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

cor_importante = ['#9dc6ae', '#bbd0ff', '#c8b6ff', '#b3dee2','#deab90']
fig, axs = plt.subplots(5, 1, figsize=(16, 18))  # Criar uma figura com 4 subplots
#fig.suptitle('Séries temporais das variáveis de interesse', fontsize=20)  # Adicionar um título à figura

# Adicionar as séries temporais aos subplots
variaveis = ['Starch Flow', 'Amina Flow', 'Ore Pulp pH', 'Ore Pulp Density','% Silica Concentrate']
for i, var in enumerate(variaveis):
    axs[i].plot(importante.index, importante[var], label=var, color=cor_importante[i])
    axs[i].set_ylabel(var)
    #axs[i].legend()
    axs[i].set_facecolor('white')  # Definir fundo branco
    decomposicao = seasonal_decompose(importante[var], model='additive', period=30, extrapolate_trend=30)
    axs[i].plot(decomposicao.trend, color='red')
    axs[i].axvline(x=pd.Timestamp('2017-08-30'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte
    axs[i].axvline(x=pd.Timestamp('2017-08-18'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte
    axs[i].axvline(x=pd.Timestamp('2017-05-10'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte

# Adicionar rótulo ao eixo x no último subplot
axs[-1].set_xlabel('Date')

# Ajustar layout e mostrar o gráfico
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar layout para não sobrepor o título
plt.savefig('Images/series_temporais_importante.png', format='png', dpi=600)

# Visualização das séries temporais
# DATAFRAME: airLevel
cor_airLevel = ['#e4e7e4', '#c0c4ca', '#9ba1b0', '#777f96', '#535c7b', '#2e3961', '#0a1647','#deab90']
fig, axs = plt.subplots(8, 1, figsize=(16, 24))  # Criar uma figura com 7 subplots
#fig.suptitle('Séries temporais das variáveis de Air Level', fontsize=20)  # Adicionar um título à figura

# Adicionar as séries temporais aos subplots
for i, coluna in enumerate(airLevel.columns):
    axs[i].plot(airLevel.index, airLevel[coluna], label=coluna, color=cor_airLevel[i])
    axs[i].set_ylabel(coluna)
   # axs[i].legend()
    axs[i].set_facecolor('white')  # Definir fundo branco
    decomposicao = seasonal_decompose(airLevel[coluna], model='additive', period=30, extrapolate_trend=30)
    axs[i].plot(decomposicao.trend, color='red')
    #axs[i].axvline(x=pd.Timestamp('2017-08-30'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte
    #axs[i].axvline(x=pd.Timestamp('2017-08-18'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte
    #axs[i].axvline(x=pd.Timestamp('2017-05-10'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte

# Adicionar rótulo ao eixo x no último subplot
axs[-1].set_xlabel('Date')

# Ajustar layout e mostrar o gráfico
plt.tight_layout()  # Ajustar layout para não sobrepor o título
plt.savefig('Images/series_temporais_airLevel.png', format='png', dpi=600)


# Visualização das séries temporais
# DATAFRAME: airFlow
cor_airFlow = ['#e4e7e4', '#c0c4ca', '#9ba1b0', '#777f96', '#535c7b', '#2e3961', '#0a1647','#deab90']
fig, axs = plt.subplots(8, 1, figsize=(16, 24))  # Criar uma figura com 7 subplots
#fig.suptitle('Séries temporais das variáveis de Air Flow', fontsize=20)  # Adicionar um título à figura

# Adicionar as séries temporais aos subplots
for i, coluna in enumerate(airFlow.columns):
    axs[i].plot(airFlow.index, airFlow[coluna], label=coluna, color=cor_airFlow[i])
    axs[i].set_ylabel(coluna)
    #axs[i].legend()
    axs[i].set_facecolor('white')  # Definir fundo branco
    decomposicao = seasonal_decompose(airFlow[coluna], model='additive', period=30, extrapolate_trend=30)
    axs[i].plot(decomposicao.trend, color='red')
    #axs[i].axvline(x=pd.Timestamp('2017-08-30'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte
    #axs[i].axvline(x=pd.Timestamp('2017-08-18'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte
    #axs[i].axvline(x=pd.Timestamp('2017-05-10'), color='black', linestyle='-',linewidth=2)  # Adicionar uma linha vertical para destacar a data de corte

# Adicionar rótulo ao eixo x no último subplot
axs[-1].set_xlabel('Date')

# Ajustar layout e mostrar o gráfico
plt.tight_layout()  # Ajustar layout para não sobrepor o título
plt.savefig('Images/series_temporais_airFlow.png', format='png', dpi=600)

# Uma analise visual rápida das series temporais indicam que, de certa maneira, 
# O pH baixo da polpa de minério, a redução de amido e amida está associado a um aumento na concentração de sílica
# Controlar esses parâmetros pode ser uma maneira de controlar a concentração de sílica

# O controle do níveis de fluxo ar nas colunas de flotação finais pode contribuir para a redução da silica



# Importação de bibliotecas

# Manipulação e tratamento de dados
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo

# Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
file_path= "base/MiningProcess_Flotation_Plant_Database.csv"
try:
    dados = pd.read_csv(
        file_path,
        decimal=',',
    )
    print(f"Arquivo carregado com sucesso: {file_path}")
except FileNotFoundError:
    print(f"Arquivo não encontrado: {file_path}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Arquivo vazio: {file_path}")
    exit(1)
except pd.errors.ParserError:
    print(f"Erro ao carregar o arquivo: {file_path}")
    exit(1)
    

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

dados_tratados= dados.groupby(dados.index).mean().dropna() # Agrupar os dados pela média diária
#dados_tratados = dados[dados.index > '2017-03-09'] # Selecionar os dados a partir de 2017-03-29

# Salvar os dados tratados em um novo arquivo CSV
dados_tratados.to_csv('base/dados_tratados_media.csv', index=True) 
dados.to_csv('base/dados_tratados.csv', index=True) 


# EDA - Análise Exploratória de Dados

# Análise de correlação
matriz_correlacao = dados_tratados.corr()

mask = np.triu(np.ones_like(matriz_correlacao, dtype=bool))# Criar uma máscara para o triângulo superior
plt.figure(figsize=(20, 20))
sns.set(font_scale=1.0)
heatmap = sns.heatmap(matriz_correlacao, annot=True, cmap='RdYlBu', 
                      fmt=".2f",  # Exibir valores com duas casas decimais
                      center=0,
                      xticklabels=matriz_correlacao.columns,  # Rótulos do eixo x
                      yticklabels=matriz_correlacao.columns,  # Rótulos do eixo y
                      annot_kws={"size": 15},  # Tamanho da fonte dentro das células
                      linewidths=0.5, linecolor='white',  # Bordas das células
                      mask=mask,
                      vmin=-1, vmax=1,  # Escala de cores de -1 a 1
                      cbar_kws={"shrink": 0.8})  # Ajustar o tamanho da barra de cores

heatmap.set_facecolor('#f8f9fa')# Definir fundo 
ax = heatmap.axes# Obter os eixos atuais
ax.set_aspect("auto")# Ajustar o aspecto dos eixos para 'auto' para ajustar automaticamente o tamanho das células
plt.tight_layout()
#plt.title("Mapa de correlação entre as variáveis")# Adicionar um título ao heatmap
plt.savefig('Images/mapa_de_correlacao.png', dpi=300)

# Alta correlação entre algumas variáveis, o que pode indicar multicolinearidade
# a correlação entre a concentração inicial e final de ferro e sílica apresentam valores negativos próximos de -1
# A correlação entre nivel da coluna de flotação e fluxo de ar apresenta valores positivos 
# Número muito grande de variáveis, o que dificulta a análise, executar analise de componentes principais para redução de dimensionalidade

# APlicação da técnica PCA para redução de dimensionalidade

x = dados_tratados
scaler = StandardScaler() # Padronização dos dados, os dados possuem escalas diferentes a padronização é necessária
x_scaled = scaler.fit_transform(x)

# Calcular a estatística KMO  > 0.6 é considerado adequado para aplicar PCA
kmo_all, kmo_model = calculate_kmo(x_scaled)
print("Estatística KMO para cada variável:\n", kmo_all)
print("Estatística KMO geral:", kmo_model)


# Aplicação da PCA sem especificar o número de componentes
pca = PCA() # para definir o número de componentes n_components=2
pca.fit(x_scaled)

# Variância explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_

# Configuração do gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Valores do eixo x
x_values = range(1, len(explained_variance) + 1)

# Criação do gráfico de barras com cores diferentes
bars = ax.bar(x_values, explained_variance, alpha=0.7)
for bar, x_val in zip(bars, x_values):
    if x_val <= 10:
        bar.set_color('#011638')
    else:
        bar.set_color('#c6c7c4')

ax.set_title('Variância Explicada por Componente Principal', loc='left', fontsize=16, pad=20, color = '#353b3c')
ax.set_xlabel('Componente Principal')
ax.set_ylabel('Variância Explicada')
ax.set_facecolor('white')  # Definir fundo branco
ax.spines['top'].set_visible(False)  # Remover a linha superior
ax.spines['right'].set_visible(False)  # Remover a linha direita

# Adicionando texto explicativo
ax.text(x=4.5, y=0.2, s="A variância explicada $\\bf{diminui}$ à medida que mais \n"
        "componentes são analisados, sugerindo que os primeiros  \n"
        "capturam a $\\bf{maior}$ parte da informação dos dados.",
        verticalalignment='top',
        fontsize=18, color='#353b3c')

# Ajustes finais e salvando a imagem
plt.tight_layout()
plt.savefig('Images/explained_variance_bar.png')


# Configuração do gráfico para variancia explicada acumulada
fig, ax1 = plt.subplots(figsize=(10, 6))
cumulative_explained_variance = explained_variance.cumsum()

# Gráfico de barras da variância explicada por cada componente principal
bars = ax1.bar(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, alpha=0.7)

# Alterar a cor das barras com base no valor
for bar, val in zip(bars, cumulative_explained_variance):
    if val < 0.9:
        bar.set_color('#011638')
    else:
        bar.set_color('#c6c7c4')

# Configuração do gráfico
ax1.set_title('Variância Explicada Acumulada para o conjunto de dados indica que', loc='left', fontsize=16, pad=20, color = '#353b3c')
ax1.set_xlabel('Componente Principal')
ax1.set_ylabel('Variância Explicada Acumulada')
ax1.axhline(y=0.9, color='r', linestyle='--')  # Linha de referência para 85% da variância explicada
ax1.set_facecolor('white')  # Definir fundo branco
ax1.spines['top'].set_visible(False)  # Remover a linha superior
ax1.spines['right'].set_visible(False)  # Remover a linha direita

# Adicionando texto explicativo
ax1.text(x=0.5, y=1, s="Até a componente PC-10 \n"
        "cerca de $\\bf{90\\%}$ da variância é preservada",
        verticalalignment='top',
        fontsize=14, color='#353b3c')

# Ajustes finais e salvando a imagem
plt.tight_layout()
plt.savefig('Images/cumulative_explained_variance_bar.png')

# Identificar a maior contribuição para cada componente principal
components = pca.components_
for i, component in enumerate(components):
    max_contrib_index = component.argmax()  # Índice da variável com maior contribuição
    max_contrib_variable = dados_tratados.columns[max_contrib_index]  # Nome da variável
    max_contrib_value = component[max_contrib_index]  # Valor da contribuição
    print(f"Componente Principal {i+1}:")
    print(f"  Variável com maior contribuição: {max_contrib_variable}")
    print(f"  Valor da contribuição: {max_contrib_value}")


# Determinar o número ótimo de componentes principais
optimal_components = next(i for i, cumulative_variance in enumerate(cumulative_explained_variance) if cumulative_variance >= 0.88) + 1
print(f'O número ótimo de componentes principais é: {optimal_components}')

# Aplicação da PCA com o número ótimo de componentes
pca_optimal = PCA(n_components=optimal_components)
principal_components = pca_optimal.fit_transform(x_scaled)

# Criação de um DataFrame com os componentes principais
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(optimal_components)])
pca_df.index = dados_tratados.index  # Manter o índice de tempo original
print(pca_df.head())
# Análise das cargas dos componentes principais
loadings = pd.DataFrame(pca_optimal.components_.T, columns=[f'PC{i+1}' for i in range(optimal_components)], index=dados_tratados.columns)
 
# Imprimir os 5 maiores loadings em módulo para cada componente principal
# for col in loadings.columns:
#     print(f"Top 5 loadings for {col}:")
#     top_loadings = loadings[col].abs().nlargest(5)
#     print(loadings.loc[top_loadings.index, col])
    

# Visualização das cargas dos componentes principais
plt.figure(figsize=(16, 8))
sns.heatmap(loadings, annot=True, cmap='RdYlBu', fmt=".2f")
#plt.title('Heatmap of Principal Component Loadings')
plt.xlabel('Componentes Principais')
#plt.ylabel('Features')
plt.tight_layout()
plt.savefig('Images/pca_loadings_heatmap.png')

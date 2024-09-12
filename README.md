# Tópicos Avançados em Aprendizado de Máquina e Otimização - Trabalho 01

Este projeto é uma aplicação de classificação de tumores utilizando redes neurais com TensorFlow e Keras. O objetivo é identificar se um tumor é benigno ou maligno com base no conjunto de dados do câncer de mama de Wisconsin.

## Descrição

O projeto inclui a construção, treinamento e avaliação de um modelo de rede neural para classificar tumores como benignos ou malignos. Utiliza-se o conjunto de dados de câncer de mama de Wisconsin para treinar o modelo e avaliar seu desempenho com base em métricas como acurácia e matriz de confusão.

## Estrutura do Projeto

O projeto está estruturado da seguinte forma:

- `main.py`: Script principal que carrega os dados, treina o modelo e avalia seu desempenho.
- `load_data.py`: Módulo responsável por carregar e processar os dados.
- `model.py`: Define a arquitetura do modelo de rede neural.
- `train.py`: Funções para treinamento do modelo, incluindo callbacks para monitoramento.
- `evaluate.py`: Função para avaliação do modelo, incluindo cálculo da acurácia e matriz de confusão.
- `epochsaver.py`: Callback personalizado para salvar a época com a melhor acurácia de validação.

## Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/hebertbruno/PGENE-613-Trabalho-01.git
    cd PGENE-613-Trabalho-01
    ```

2. Crie um ambiente virtual (opcional, mas recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows `venv\Scripts\activate`
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Execute o script principal para iniciar o treinamento e a avaliação do modelo:

    ```bash
    python main.py
    ```

2. Você será solicitado a escolher uma condição para treinar o modelo. As opções disponíveis são:

    - `a`: Sem regularização e sem dropout, sem parada antecipada
    - `b`: Sem regularização, sem dropout, com parada antecipada
    - `c`: Com regularização L2 nas duas camadas, sem dropout
    - `d`: Com regularização L1 nas duas camadas, sem dropout
    - `e`: Sem regularização, com dropout de 30% na segunda camada
    - `f`: Com regularização L2 nas duas camadas e dropout de 30%

## Arquivos Gerados

Durante a execução do treinamento, o seguinte arquivo será gerado:

- `/tmp/ckpt/checkpoint.model.keras`: Melhor modelo salvo se a condição de treinamento especificar a não utilização de parada antecipada.

## Dependências

O projeto utiliza as seguintes bibliotecas:

- TensorFlow
- Matplotlib
- NumPy
- Pandas
- scikit-learn
- ucimlrepo



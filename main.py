from load_data import load_and_process_data
from model import build_model
from train import train_model
from evaluate import evaluate_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1, l2
import matplotlib.pyplot as plt
import os
import numpy as np


def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_history(history, condicao):
    # Plotar a acurácia
    plt.figure()
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title(f'Acurácia do Modelo na condicao {condicao}')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend()
    plt.savefig(f'acuracia_Condicao_{condicao}.png')  # Salva o gráfico em um arquivo
    plt.close()

def main():
    # Carregar e processar os dados
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_process_data()
    print("Número de amostras em X_train:", X_train.shape[0])
    print("Número de amostras em X_val:", X_val.shape[0])
    print("Número de amostras em X_test:", X_test.shape[0])
    benign_count = np.sum(y_test == 0)  # Contar quantos são benignos (0)
    malignant_count = np.sum(y_test == 1)  # Contar quantos são malignos (1)

    print(f"Total de amostras benignas (0): {benign_count}")
    print(f"Total de amostras malignas (1): {malignant_count}")

    print("Escolha uma das condições abaixo para treinar o modelo:")
    print("a: Sem regularização e sem dropout, sem early stopping")
    print("b: Sem regularização, sem dropout, com early stopping")
    print("c: Com regularização L2 nas duas camadas, sem dropout")
    print("d: Com regularização L1 nas duas camadas, sem dropout")
    print("e: Sem regularização, com dropout de 30% na segunda camada")
    print("f: Com regularização L2 nas duas camadas e dropout de 30%")
    condicao = input("Digite a letra da condição escolhida (a, b, c, d, e, f): ").lower()
    # Converter as labels para one-hot encoding
    y_train_categorical = to_categorical(y_train, num_classes=2)
    y_val_categorical = to_categorical(y_val, num_classes=2)
    y_test_categorical = to_categorical(y_test, num_classes=2)

    # Escolha uma das condições abaixo para treinar seu modelo:
    if condicao == 'a':
        model = build_model(input_shape=X_train.shape[1:], regularization=None, dropout_rate=0.0)
        history = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, use_early_stopping=False)
    elif condicao == 'b':
        model = build_model(input_shape=X_train.shape[1:], regularization=None, dropout_rate=0.0)
        history = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, use_early_stopping=True)
    elif condicao == 'c':
        model = build_model(input_shape=X_train.shape[1:], regularization=l2(0.01), dropout_rate=0.0)
        history = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, use_early_stopping=False)
    elif condicao == 'd':
        model = build_model(input_shape=X_train.shape[1:], regularization=l1(0.01), dropout_rate=0.0)
        history = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, use_early_stopping=False)
    elif condicao == 'e':
        model = build_model(input_shape=X_train.shape[1:], regularization=None, dropout_rate=0.3)
        history = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, use_early_stopping=False)
    elif condicao == 'f':
        model = build_model(input_shape=X_train.shape[1:], regularization=l2(0.01), dropout_rate=0.3)
        history = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, use_early_stopping=False)
    else:
        print("Opção inválida. Tente novamente.")
    
    # Carregar o melhor modelo salvo
    checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
    ensure_directory_exists(checkpoint_filepath)
    best_model = load_model(checkpoint_filepath)

    # Avaliar o modelo
    accuracy, cm = evaluate_model(best_model, X_test, y_test_categorical)

    print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
    print("Matriz de Confusão:")
    print(cm)
    #plot_history(history, condicao)

if __name__ == "__main__":
    main()

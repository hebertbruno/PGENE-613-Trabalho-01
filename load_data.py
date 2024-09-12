from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_process_data():
    # Buscar e carregar o conjunto de dados
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # Certificar-se de que y é uma Série
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()  # Converte para Série, se for DataFrame

    # Converter os valores de saída para inteiros (M: Maligno, B: Benigno)
    y = y.map({'M': 1, 'B': 0}).astype('int32')

    # Dividir os dados em conjunto de treinamento e conjunto de teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Dentro dos dados de treinamento, dividir em treinamento e validação (80/20 do treinamento)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

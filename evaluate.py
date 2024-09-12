from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test):
    # Fazer previsões no conjunto de teste
    y_pred_proba = model.predict(X_test)
    
    # Converter as previsões de probabilidade para rótulos de classe (classe com maior probabilidade)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Converter y_test de one-hot encoding para rótulos de classe
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Obtendo o valor da acurácia
    accuracy = accuracy_score(y_test_labels, y_pred)
    
    # Gerando a matriz de confusão
    cm = confusion_matrix(y_test_labels, y_pred)
    
    return accuracy, cm

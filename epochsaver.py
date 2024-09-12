from tensorflow.keras.callbacks import Callback

# A classe EpochSaver herda da classe Callback do Keras. Ela é usada para salvar
# a melhor época (epoch) com base na acurácia de validação ('val_accuracy') durante o treinamento.
class EpochSaver(Callback):
    
    # O método __init__ é o construtor da classe. Aqui, inicializamos a época 
    # e a melhor acurácia de validação com valores padrão (zero).
    def __init__(self):
        super().__init__()  # Chama o construtor da classe pai (Callback)
        self.best_epoch = 0  # Variável para armazenar a melhor época
        self.best_val_accuracy = 0  # Variável para armazenar a melhor acurácia de validação

    # Este método é executado no final de cada época durante o treinamento.
    # Ele verifica a acurácia de validação e, se a nova acurácia for melhor,
    # atualiza o valor da melhor acurácia e da melhor época.
    def on_epoch_end(self, epoch, logs=None):
        # Pega a acurácia de validação da época atual a partir dos logs (fornecidos pelo Keras).
        current_val_accuracy = logs.get('val_accuracy')
        
        # Se a acurácia de validação atual for maior do que a melhor acurácia armazenada,
        # atualizamos a melhor acurácia e a melhor época.
        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy  # Atualiza a melhor acurácia
            self.best_epoch = epoch + 1  # Atualiza a melhor época (incrementa por 1, pois epoch começa em 0)

            

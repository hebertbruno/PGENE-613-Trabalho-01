from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2

def build_model(input_shape, regularization=None, dropout_rate=0.0):
    model = Sequential()

    # Primeira camada oculta com regularização e ativação 'relu'
    model.add(Dense(16, activation='relu', input_shape=input_shape, kernel_regularizer=regularization))


    # Segunda camada oculta com regularização e ativação 'relu'
    model.add(Dense(8, activation='relu', kernel_regularizer=regularization))

    # Dropout opcional após a segunda camada
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Camada de saída para classificação binária (probabilidade da amostra de ser benigna ou maligna)
    model.add(Dense(2, activation='softmax'))

    # Compilando o modelo com Adam e função de perda categórica
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from epochsaver import EpochSaver  # Importe o callback customizado
import os

os.makedirs('/tmp/ckpt', exist_ok=True)

def train_model(model, X_train, y_train, X_val, y_val, max_epochs=100, patience=3, use_early_stopping=True):
    
    callbacks = []

    # Checkpoint para salvar o melhor modelo, adicionado somente se early stopping estiver desativado
    if not use_early_stopping:
        checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        callbacks.append(model_checkpoint_callback)

    # Callback para salvar a época do melhor modelo
    epoch_saver = EpochSaver()
    callbacks.append(epoch_saver)

    # Adicionar parada antecipada se especificado
    if use_early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping_callback)

    # Treinamento do modelo
    history = model.fit(
        X_train, y_train,
        epochs=max_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Salvar a época do melhor modelo
    best_epoch = epoch_saver.best_epoch
    print(f"A época do melhor modelo é: {best_epoch}")

    # Obter a val_loss e val_accuracy do modelo usado para teste
    if use_early_stopping:
        # O modelo usado para teste é o último modelo antes do loss aumentar por 3 epocas
        last_epoch = len(history.history['accuracy']) - patience
    else:
        # O modelo usado para teste é o modelo salvo pelo checkpoint, que pode ser o melhor até o final do treinamento
        last_epoch = best_epoch

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Época do modelo utilizado para teste: {last_epoch}")
    print(f"Modelo de Validação Escolhido - val_accuracy: {val_accuracy:.4f}, val_loss: {val_loss:.4f}")

    return history

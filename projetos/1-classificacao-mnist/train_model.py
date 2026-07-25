import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Garante que os artefatos sejam salvos sempre na pasta deste script,
# independentemente do diretório a partir do qual ele é executado.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model.h5")

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

# 1. Carregamento do dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalização para [0, 1] e ajuste do shape (28, 28) -> (28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"Treino: {x_train.shape} | Teste: {x_test.shape}")

# 3. O split treino/validação é feito via validation_split no model.fit (abaixo)

# 4. Construção da CNN com 4 blocos Conv2D + BatchNormalization + MaxPooling2D
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Bloco 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco 4
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding="same"),

    # Classificador
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 5. Treinamento com EarlyStopping monitorando val_loss
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=2,
)

# 6. Exibição da acurácia de validação final no terminal
# Importante: restore_best_weights=True restaura os pesos da época com MENOR
# val_loss (não a de maior val_accuracy). Para reportar a acurácia condizente
# com o modelo que de fato foi salvo, buscamos o val_accuracy dessa mesma época.
best_epoch = int(np.argmin(history.history["val_loss"]))
best_val_acc = history.history["val_accuracy"][best_epoch]
print(f"\nMelhor época (menor val_loss): {best_epoch + 1}")
print(f"Acurácia de validação final: {best_val_acc:.4f}")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia no conjunto de teste: {test_acc:.4f}")

# 7. Salvamento do modelo treinado
model.save(MODEL_PATH)
print(f"\nModelo salvo em {MODEL_PATH}")
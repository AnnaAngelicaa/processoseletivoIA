import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

# Garante que os artefatos sejam lidos/salvos sempre na pasta deste script,
# independentemente do diretório a partir do qual ele é executado.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_H5_PATH = os.path.join(SCRIPT_DIR, "model.h5")
MODEL_TFLITE_PATH = os.path.join(SCRIPT_DIR, "model.tflite")

# 1. Carregamento do modelo treinado em "model.h5"
print(f"Carregando modelo de {MODEL_H5_PATH}...")
model = tf.keras.models.load_model(MODEL_H5_PATH)

# 2. Conversão para TensorFlow Lite usando tf.lite.TFLiteConverter, com
#    3. Dynamic Range Quantization aplicada via optimizations = [DEFAULT].
#    Essa técnica quantiza apenas os pesos (float32 -> int8); as ativações
#    permanecem em float32 e a conversão é feita dinamicamente durante a
#    inferência, por isso não exige um dataset representativo (diferente
#    da quantização full-integer).
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Convertendo para TensorFlow Lite com Dynamic Range Quantization...")
tflite_model = converter.convert()

# 4. Salvar o resultado como "model.tflite"
with open(MODEL_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

# Comparação de tamanhos, útil para o relatório
h5_size_kb = os.path.getsize(MODEL_H5_PATH) / 1024
tflite_size_kb = os.path.getsize(MODEL_TFLITE_PATH) / 1024

print(f"\nModelo original (model.h5):      {h5_size_kb:.1f} KB")
print(f"Modelo otimizado (model.tflite): {tflite_size_kb:.1f} KB")
print(f"Redução de tamanho: {(1 - tflite_size_kb / h5_size_kb) * 100:.1f}%")
print(f"\nModelo TFLite salvo em {MODEL_TFLITE_PATH}")
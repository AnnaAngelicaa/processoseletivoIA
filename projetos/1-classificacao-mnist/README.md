## 📝 Relatório do Candidato

👤 **Nome Completo:** Anna Angélica Costa de Souza

### 1️⃣ Resumo da Arquitetura do Modelo

CNN com **4 blocos convolucionais** (`Conv2D → BatchNormalization → MaxPooling2D`), com filtros crescentes: 32 → 64 → 128 → 128. Após o `Flatten`, uma camada densa de 128 unidades (ReLU) seguida de **Dropout(0.5)** antes da saída (Dense 10, softmax). Total: 308.618 parâmetros.

Validação via `validation_split=0.2` no `model.fit` (batch_size=128, até 15 épocas). `EarlyStopping` monitora `val_loss` (`patience=3`, `restore_best_weights=True`).

**Justificativa:** Dropout de 0.5 (mais alto que o usual) foi escolhido porque atua só na camada densa final — a parte com maior risco de overfitting, já que as camadas convolucionais já são regularizadas pelo BatchNorm e pelo compartilhamento de pesos.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow 2.21.0 · NumPy 1.26.4 · Python 3.11

### 3️⃣ Técnica de Otimização do Modelo

**Dynamic Range Quantization**, via `converter.optimizations = [tf.lite.Optimize.DEFAULT]`. Quantiza os pesos de `float32` para `int8`; as ativações permanecem em `float32` e a conversão ocorre dinamicamente na inferência — não exige dataset representativo.

### 4️⃣ Resultados Obtidos

- Acurácia de validação: **98.67%** (época com menor `val_loss`)
- Acurácia de teste: **98.67%**
- `model.h5`: 3700.6 KB → `model.tflite`: 318.8 KB
- **Redução de 91.4%**

### 5️⃣ Comentários Adicionais (Opcional)

Maior dificuldade foi o ambiente: instalações corrompidas de NumPy/TensorFlow por instabilidade de rede no Codespace (download de ~570MB truncando sem erro visível), resolvido com `pip install --force-reinstall --no-cache-dir --default-timeout=120 --retries 5`.

Decisão técnica: corrigi o cálculo da acurácia final para usar `argmin(val_loss)` em vez de `max(val_accuracy)`, já que `restore_best_weights` restaura pela menor perda, não pela maior acurácia — nem sempre coincidem.

Limitação: Dynamic Range Quantization reduz tamanho em disco, mas o ganho de velocidade é menor que quantização full-integer (ainda há conversão float32↔int8 em runtime).

### 6️⃣ Exemplo de Inferência

```
Amostra 1: predito=7 | real=7 ✓
Amostra 2: predito=2 | real=2 ✓
Amostra 3: predito=1 | real=1 ✓
Amostra 4: predito=0 | real=0 ✓
Amostra 5: predito=4 | real=4 ✓

Acertos: 5/5
```

Todas as 5 amostras acertaram, cobrindo dígitos variados. Resultado coerente com a acurácia de 98.67% — a quantização não introduziu degradação perceptível nessas amostras.

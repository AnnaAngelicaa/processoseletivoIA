## 📝 Relatório do Candidato

👤 **Nome Completo:** Anna Angélica Costa de Souza

### 1️⃣ Resumo da Arquitetura do Modelo

CNN com **4 blocos convolucionais** (`Conv2D → BatchNormalization → MaxPooling2D`), com filtros crescentes: 32 → 64 → 128 → 128. Após o `Flatten`, uma camada densa de 128 unidades (ReLU) seguida de **Dropout(0.5)** antes da saída (Dense 10, softmax). Total: 308.618 parâmetros.

Validação via `validation_split=0.2` no `model.fit` (batch_size=128, até 15 épocas). `EarlyStopping` monitora `val_loss` (`patience=3`, `restore_best_weights=True`).

**Justificativa:** Dropout de 0.5 (mais alto que o usual) foi escolhido porque atua só na camada densa final — a parte com maior risco de overfitting, já que as camadas convolucionais já são regularizadas pelo BatchNorm e pelo compartilhamento de pesos.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow 2.21.0 · Keras 3.12.3 · NumPy 2.2.6 · Python 3.11

### 3️⃣ Técnica de Otimização do Modelo

**Dynamic Range Quantization**, via `converter.optimizations = [tf.lite.Optimize.DEFAULT]`. Quantiza os pesos de `float32` para `int8`; as ativações permanecem em `float32` e a conversão ocorre dinamicamente na inferência — não exige dataset representativo.

### 4️⃣ Resultados Obtidos

- Acurácia de validação: **99.10%** (época 4, menor `val_loss`)
- Acurácia de teste: **99.15%**
- `model.h5`: 3699.5 KB → `model.tflite`: 318.8 KB
- **Redução de 91.4%**

### 5️⃣ Comentários Adicionais (Opcional)

A maior dificuldade não foi de modelagem, mas de reprodutibilidade de ambiente. O `requirements.txt` do desafio não fixa versões de `keras`/`numpy` (dependências transitivas do `tensorflow`), então o `model.h5` treinado localmente (Keras 3.15.0) falhava ao ser carregado pelo pipeline de CI, que resolvia para Keras 3.12.3 — uma versão anterior que não reconhece o formato de serialização mais novo do inicializador `GlorotUniform` (`Error when deserializing class 'Conv2D'`). Diagnostiquei comparando os logs de instalação do CI com `pip show` local, e resolvi recriando um ambiente virtual com as mesmas versões exatas do CI (`tensorflow==2.21.0`, `keras==3.12.3`, `numpy==2.2.6`) e regerando `model.h5`/`model.tflite` nele, sem alterar o `requirements.txt` original do projeto.

Decisão técnica adicional: corrigi o cálculo da acurácia final em `train_model.py` para usar `argmin(val_loss)` em vez de `max(val_accuracy)`, já que `restore_best_weights=True` restaura os pesos da época com menor perda de validação — não necessariamente a de maior acurácia.

Limitação: Dynamic Range Quantization reduz bem o tamanho em disco, mas o ganho de velocidade de inferência é menor do que se obteria com quantização full-integer, já que ainda há conversão float32↔int8 em tempo de execução.

### 6️⃣ Exemplo de Inferência

```
Rodando inferencia em 5 amostras usando model.tflite:
Amostra 1: predito=7 | real=7 ✓
Amostra 2: predito=2 | real=2 ✓
Amostra 3: predito=1 | real=1 ✓
Amostra 4: predito=0 | real=0 ✓
Amostra 5: predito=4 | real=4 ✓
Acertos: 5/5
```

Todas as 5 amostras acertaram, cobrindo dígitos variados (7, 2, 1, 0, 4). Resultado coerente com a acurácia de teste de 99.15% — a quantização dinâmica não introduziu degradação perceptível nessas amostras, confirmando que o artefato de edge (`model.tflite`) preserva o comportamento do modelo original mesmo após a redução de 91.4% no tamanho do arquivo.
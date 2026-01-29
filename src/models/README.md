# RoBERTa Fake News Classifier

Scripts para treinamento e predição de modelo RoBERTa (BERTimbau) para detecção de fake news.

## 📋 Descrição

Este módulo implementa um classificador de deep learning usando o modelo BERTimbau (BERT pré-treinado em português) para detectar notícias falsas no dataset FakeBr News.

## 🚀 Scripts Disponíveis

### 1. `train_roberta_classifier.py`

Script principal para treinar o modelo RoBERTa.

**Uso básico:**
```bash
python src/models/train_roberta_classifier.py
```

**Parâmetros disponíveis:**
- `--data-path`: Caminho para o arquivo de dados (padrão: `data/processed/fakebr_news.parquet`)
- `--model-name`: Nome do modelo pré-treinado (padrão: `neuralmind/bert-base-portuguese-cased`)
- `--max-length`: Comprimento máximo das sequências (padrão: `128`)
- `--batch-size`: Tamanho do batch (padrão: `8`)
- `--epochs`: Número de épocas (padrão: `3`)
- `--learning-rate`: Taxa de aprendizado (padrão: `2e-5`)
- `--test-size`: Proporção do conjunto de teste (padrão: `0.2`)
- `--random-state`: Seed para reprodutibilidade (padrão: `42`)
- `--save-dir`: Diretório para salvar modelos e resultados (padrão: `models`)

**Exemplo com parâmetros customizados:**
```bash
python src/models/train_roberta_classifier.py \
    --data-path data/processed/fakebr_news.parquet \
    --epochs 5 \
    --batch-size 16 \
    --max-length 256 \
    --save-dir models/roberta_v1
```

**Saídas geradas:**
- `roberta_best_model.bin`: Modelo treinado com melhor acurácia
- `config.json`: Configuração utilizada no treinamento
- `metrics.json`: Métricas de avaliação final
- `training_history.json`: Histórico de treinamento (loss e accuracy por época)
- `training_history.png`: Gráficos de loss e accuracy
- `confusion_matrix.png`: Matriz de confusão

### 2. `predict_roberta.py`

Script para fazer predições com modelo treinado.

**Uso para texto único:**
```bash
python src/models/predict_roberta.py \
    --model-dir models \
    --text "Texto da notícia para classificar"
```

**Uso para arquivo:**
```bash
python src/models/predict_roberta.py \
    --model-dir models \
    --input-file data/processed/news_to_classify.parquet \
    --text-column preprocessed_text \
    --output-file results/predictions.parquet
```

**Parâmetros disponíveis:**
- `--model-dir`: Diretório contendo o modelo treinado (padrão: `models`)
- `--text`: Texto único para classificar
- `--input-file`: Arquivo parquet com textos para classificar
- `--text-column`: Nome da coluna com textos (padrão: `preprocessed_text`)
- `--output-file`: Arquivo para salvar resultados

## 📊 Arquitetura do Modelo

- **Base**: BERTimbau (neuralmind/bert-base-portuguese-cased)
- **Tipo**: Sequence Classification
- **Classes**: 2 (True News, Fake News)
- **Parâmetros**: ~109 milhões

## 🔧 Técnicas Utilizadas

1. **Fine-tuning**: Ajuste fino do modelo pré-treinado
2. **Gradient Clipping**: Normalização de gradientes (max_norm=1.0)
3. **Learning Rate Scheduling**: Warmup linear
4. **Stratified Split**: Divisão estratificada mantendo pares de notícias
5. **Early Stopping**: Salvamento do melhor modelo baseado em acurácia de validação

## 📈 Métricas de Avaliação

O modelo é avaliado usando:
- Acurácia
- F1-Score
- Precision
- Recall
- Matriz de Confusão
- Classification Report

## 💡 Exemplos de Uso

### Treinar modelo básico
```bash
python src/models/train_roberta_classifier.py
```

### Treinar com GPU e mais épocas
```bash
python src/models/train_roberta_classifier.py \
    --epochs 5 \
    --batch-size 32
```

### Fazer predição em texto
```bash
python src/models/predict_roberta.py \
    --text "Presidente anuncia nova medida econômica"
```

### Classificar arquivo completo
```bash
python src/models/predict_roberta.py \
    --input-file data/processed/party_news.parquet \
    --output-file results/party_news_classified.parquet
```

## 📦 Dependências

```
torch>=2.0.0
transformers>=4.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## 🎯 Resultados Esperados

Com as configurações padrão, espera-se:
- Acurácia de validação: ~85-95%
- F1-Score: ~0.85-0.95
- Tempo de treinamento: ~2-4 horas (CPU) ou ~30-60 min (GPU)

## 📝 Notas

- O treinamento em CPU pode ser muito lento. Recomenda-se usar GPU se disponível.
- O modelo salvo pode ocupar ~400-500 MB de espaço em disco.
- Para melhores resultados, considere aumentar `max_length` para 256 ou 512.
- Batch size maior requer mais memória mas pode acelerar o treinamento.

## 🔍 Troubleshooting

**Erro de memória (CUDA out of memory):**
- Reduza `batch_size` para 4 ou 2
- Reduza `max_length` para 64 ou 128

**Treinamento muito lento:**
- Verifique se está usando GPU: `torch.cuda.is_available()`
- Reduza o tamanho do dataset para testes
- Reduza número de épocas

**Modelo não converge:**
- Aumente o número de épocas
- Ajuste a taxa de aprendizado (tente 1e-5 ou 5e-5)
- Verifique se os dados estão balanceados

# Fake News Detector - Brazilian Politics

## 📋 Sobre o Projeto

Trabalho de Conclusão de Curso (TCC) desenvolvido na **USP ESALQ** (2026) para a disciplina de **Data Science e Analytics**.

Este projeto aplica modelos de Aprendizado de Máquina para detectar notícias falsas (fake news) no contexto político brasileiro, utilizando o dataset **FakeNews.Br** que contém 7.200 notícias (3.600 falsas e 3.600 verdadeiras).

## 🎯 Objetivo

Desenvolver e comparar diferentes modelos de Machine Learning e Deep Learning para classificação automática de notícias como verdadeiras ou falsas, contribuindo para o combate à desinformação no cenário político brasileiro.

## 📊 Dataset

- **Total de notícias**: 7.200
- **Notícias falsas**: 3.600 (50%)
- **Notícias verdadeiras**: 3.600 (50%)
- **Período**: 2016-2018
- **Categorias**:
  - Política: 4.180 (58.0%)
  - TV & Celebridades: 1.544 (21.4%)
  - Sociedade & Cotidiano: 1.276 (17.7%)
  - Ciência & Tecnologia: 112 (1.5%)
  - Economia: 44 (0.7%)
  - Religião: 44 (0.7%)

## 🔬 Modelos Implementados

### Modelos de Machine Learning Clássico
1. **Support Vector Machine (SVM)**
2. **Random Forest**
3. **Naive Bayes**

### Modelos de Deep Learning
4. **RoBERTa** (Robustly Optimized BERT Pretraining Approach)

## 📁 Estrutura do Projeto

```
fake-news-detector-brazilian-politics/
│
├── data/
│   ├── raw/              # Dados brutos (14.400 arquivos de texto)
│   ├── processed/        # Dados processados (parquet)
│   └── external/         # Dados externos (party_news)
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_exploratory_analysis_external_data.ipynb
│   ├── 03_svm_fake_news_classifier.ipynb
│   ├── 04_random_forest_classifier.ipynb
│   ├── 05_naive_bayes_classifier.ipynb
│   └── 06_roberta_deep_learning_classifier.ipynb
│
├── src/
│   ├── data/             # Scripts de processamento de dados
│   ├── features/         # Engenharia de features
│   ├── models/           # Implementação dos modelos
│   └── evaluation/       # Avaliação e métricas
│
├── models/               # Modelos treinados salvos
│
└── README.md
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Scikit-learn** - Modelos de ML clássico
- **Transformers (HuggingFace)** - Modelo RoBERTa
- **NLTK/spaCy** - Processamento de linguagem natural
- **Matplotlib/Seaborn** - Visualização de dados
- **Jupyter Notebook** - Desenvolvimento e análise

## 📈 Features Extraídas

O projeto utiliza diversas features linguísticas e estatísticas:

- **Textuais**: Número de tokens, palavras, tipos
- **Sintáticas**: Verbos, nomes, adjetivos, advérbios, pronomes
- **Estilísticas**: Palavras em maiúscula, links internos
- **Métricas**: 
  - Tamanho médio de sentenças
  - Tamanho médio de palavras
  - Pausality (pausas no texto)
  - Emotiveness (emotividade)
  - Diversity (diversidade lexical)
  - Erros ortográficos

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas numpy scikit-learn transformers torch matplotlib seaborn jupyter
```

### Executando os Notebooks

1. Clone o repositório:
```bash
git clone https://github.com/lukslana/fake-news-detector-brazilian-politics.git
cd fake-news-detector-brazilian-politics
```

2. Inicie o Jupyter Notebook:
```bash
jupyter notebook
```

3. Navegue até a pasta `notebooks/` e execute os notebooks na ordem:
   - Comece pela análise exploratória (`01_exploratory_analysis.ipynb`)
   - Depois execute os notebooks dos modelos (03 a 06)

## 📊 Resultados

Os resultados detalhados de cada modelo, incluindo métricas de desempenho (acurácia, precisão, recall, F1-score) e comparações, estão disponíveis nos respectivos notebooks.

## 👨‍🎓 Autor

**Lucas Lana**
- GitHub: [@lukslana](https://github.com/lukslana)
- Instituição: USP ESALQ
- Curso: Data Science e Analytics
- Ano: 2026

## 📝 Licença

Este projeto foi desenvolvido para fins acadêmicos como Trabalho de Conclusão de Curso.

## 🙏 Agradecimentos

- USP ESALQ - Universidade de São Paulo, Escola Superior de Agricultura "Luiz de Queiroz"
- Criadores do dataset FakeNews.Br
- Comunidade open-source de Data Science e NLP

## 📚 Referências

- Dataset FakeNews.Br
- Documentação Scikit-learn
- Documentação HuggingFace Transformers
- Artigos científicos sobre detecção de fake news

---

**Nota**: Este projeto faz parte do Trabalho de Conclusão de Curso (TCC) da disciplina de Data Science e Analytics da USP ESALQ, 2026.

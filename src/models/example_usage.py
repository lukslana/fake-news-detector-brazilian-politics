"""
Exemplo de uso dos scripts de treinamento e predição do RoBERTa.

Este script demonstra como usar as classes RoBERTaTrainer e RoBERTaPredictor
de forma programática (sem linha de comando).

Autor: Lucas Lana
Data: 2026-01-29
"""

from train_roberta_classifier import RoBERTaTrainer
from predict_roberta import RoBERTaPredictor
import os


def exemplo_treinamento():
    """Exemplo de treinamento do modelo."""
    print("="*60)
    print("EXEMPLO: TREINAMENTO DO MODELO")
    print("="*60)
    
    # Configuração
    config = {
        'model_name': 'neuralmind/bert-base-portuguese-cased',
        'max_length': 128,
        'batch_size': 8,
        'epochs': 3,
        'learning_rate': 2e-5,
        'test_size': 0.2,
        'random_state': 42
    }
    
    # Caminhos
    data_path = '../../data/processed/fakebr_news.parquet'
    save_dir = '../../models'
    
    # Criar diretório
    os.makedirs(save_dir, exist_ok=True)
    
    # Inicializar trainer
    trainer = RoBERTaTrainer(config)
    
    # Carregar dados
    df = trainer.load_data(data_path)
    
    # Preparar dados
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Configurar modelo
    trainer.setup_model()
    
    # Criar DataLoaders
    train_loader, test_loader = trainer.create_data_loaders(
        X_train, X_test, y_train, y_test
    )
    
    # Configurar otimizador
    trainer.setup_optimizer(train_loader)
    
    # Treinar
    best_accuracy = trainer.train(train_loader, test_loader, save_dir)
    
    # Plotar histórico
    trainer.plot_training_history(save_dir)
    
    # Salvar histórico
    trainer.save_training_history(save_dir)
    
    # Avaliação final
    y_true, y_pred, metrics = trainer.evaluate_final(test_loader, save_dir)
    
    # Plotar matriz de confusão
    trainer.plot_confusion_matrix(y_true, y_pred, save_dir)
    
    print(f"\nTreinamento concluído! Melhor acurácia: {best_accuracy:.4f}")
    return trainer, metrics


def exemplo_predicao_texto():
    """Exemplo de predição para texto único."""
    print("\n" + "="*60)
    print("EXEMPLO: PREDIÇÃO PARA TEXTO ÚNICO")
    print("="*60)
    
    # Inicializar preditor
    predictor = RoBERTaPredictor(model_dir='../../models')
    
    # Texto de exemplo
    texto = """
    governo federal anuncia investimento bilionario infraestrutura pais
    presidente afirmou medida vai gerar milhoes empregos proximos anos
    especialistas economia avaliam impacto positivo crescimento
    """
    
    # Fazer predição
    pred, probs = predictor.predict(texto, return_probabilities=True)
    
    print(f"\nTexto: {texto.strip()}")
    print(f"\nPredição: {'FAKE NEWS' if pred == 1 else 'TRUE NEWS'}")
    print(f"\nProbabilidades:")
    print(f"  True News: {probs['true_news']:.4f} ({probs['true_news']*100:.2f}%)")
    print(f"  Fake News: {probs['fake_news']:.4f} ({probs['fake_news']*100:.2f}%)")
    
    return predictor


def exemplo_predicao_batch():
    """Exemplo de predição para múltiplos textos."""
    print("\n" + "="*60)
    print("EXEMPLO: PREDIÇÃO PARA MÚLTIPLOS TEXTOS")
    print("="*60)
    
    # Inicializar preditor
    predictor = RoBERTaPredictor(model_dir='../../models')
    
    # Textos de exemplo
    textos = [
        "presidente anuncia nova medida economica pais",
        "alienigenas invadem terra proxima semana afirma cientista",
        "congresso aprova projeto lei importante votacao",
        "descoberta cura cancer feita cientistas brasileiros",
        "ministro defende reforma tributaria reuniao senado"
    ]
    
    # Fazer predições
    predictions, probabilities = predictor.predict_batch(
        textos, 
        return_probabilities=True
    )
    
    # Mostrar resultados
    print("\nResultados:")
    for i, (texto, pred, probs) in enumerate(zip(textos, predictions, probabilities), 1):
        label = 'FAKE NEWS' if pred == 1 else 'TRUE NEWS'
        confidence = probs['fake_news'] if pred == 1 else probs['true_news']
        
        print(f"\n{i}. {texto}")
        print(f"   → {label} (confiança: {confidence:.2%})")
    
    return predictions, probabilities


def exemplo_predicao_arquivo():
    """Exemplo de predição para arquivo."""
    print("\n" + "="*60)
    print("EXEMPLO: PREDIÇÃO PARA ARQUIVO")
    print("="*60)
    
    import pandas as pd
    
    # Inicializar preditor
    predictor = RoBERTaPredictor(model_dir='../../models')
    
    # Carregar arquivo
    input_file = '../../data/processed/fakebr_news.parquet'
    df = pd.read_parquet(input_file)
    
    # Pegar apenas uma amostra para exemplo
    df_sample = df.sample(n=100, random_state=42)
    
    # Fazer predições
    df_result = predictor.predict_dataframe(
        df_sample,
        text_column='preprocessed_text',
        output_path='../../results/predictions_sample.parquet'
    )
    
    # Mostrar resumo
    print(f"\nTotal de textos: {len(df_result)}")
    print(f"\nDistribuição das predições:")
    print(df_result['predicted_class'].value_counts())
    
    if 'is_fake' in df_result.columns:
        df_result['correct'] = (
            df_result['predicted_label'] == df_result['is_fake'].astype(int)
        )
        accuracy = df_result['correct'].mean()
        print(f"\nAcurácia na amostra: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return df_result


def main():
    """Função principal com exemplos."""
    print("EXEMPLOS DE USO - RoBERTa Fake News Classifier")
    print("="*60)
    
    # Escolher qual exemplo executar
    print("\nEscolha um exemplo:")
    print("1. Treinamento completo do modelo")
    print("2. Predição para texto único")
    print("3. Predição para múltiplos textos")
    print("4. Predição para arquivo")
    print("5. Executar todos os exemplos de predição")
    
    escolha = input("\nDigite o número do exemplo (1-5): ").strip()
    
    if escolha == '1':
        exemplo_treinamento()
    elif escolha == '2':
        exemplo_predicao_texto()
    elif escolha == '3':
        exemplo_predicao_batch()
    elif escolha == '4':
        exemplo_predicao_arquivo()
    elif escolha == '5':
        print("\n" + "="*60)
        print("EXECUTANDO TODOS OS EXEMPLOS DE PREDIÇÃO")
        print("="*60)
        exemplo_predicao_texto()
        exemplo_predicao_batch()
        exemplo_predicao_arquivo()
    else:
        print("Opção inválida!")
    
    print("\n" + "="*60)
    print("EXEMPLOS CONCLUÍDOS!")
    print("="*60)


if __name__ == '__main__':
    main()

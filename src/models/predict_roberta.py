"""
Script para fazer predições com modelo RoBERTa treinado.

Este script carrega um modelo RoBERTa treinado e faz predições em novos textos.

Autor: Lucas Lana
Data: 2026-01-29
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json


class RoBERTaPredictor:
    """Classe para fazer predições com modelo RoBERTa."""
    
    def __init__(self, model_dir, device=None):
        """
        Inicializa o preditor.
        
        Args:
            model_dir: Diretório contendo o modelo e configuração
            device: Device para executar o modelo (cuda/cpu)
        """
        self.model_dir = model_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Carregar configuração
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # Carregar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=2
        )
        
        # Carregar pesos treinados
        model_path = os.path.join(model_dir, 'roberta_best_model.bin')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Modelo carregado de: {model_dir}")
        print(f"Device: {self.device}")
    
    def predict(self, text, return_probabilities=False):
        """
        Faz predição para um texto.
        
        Args:
            text: Texto para classificar
            return_probabilities: Se True, retorna probabilidades
            
        Returns:
            Predição (0=True News, 1=Fake News) e opcionalmente probabilidades
        """
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            _, prediction = torch.max(logits, dim=1)
        
        pred_label = prediction.item()
        
        if return_probabilities:
            probs = probabilities[0].cpu().numpy()
            return pred_label, {
                'true_news': float(probs[0]),
                'fake_news': float(probs[1])
            }
        
        return pred_label
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Faz predições para múltiplos textos.
        
        Args:
            texts: Lista de textos para classificar
            return_probabilities: Se True, retorna probabilidades
            
        Returns:
            Lista de predições e opcionalmente probabilidades
        """
        predictions = []
        probabilities = [] if return_probabilities else None
        
        for text in texts:
            if return_probabilities:
                pred, probs = self.predict(text, return_probabilities=True)
                predictions.append(pred)
                probabilities.append(probs)
            else:
                pred = self.predict(text, return_probabilities=False)
                predictions.append(pred)
        
        if return_probabilities:
            return predictions, probabilities
        
        return predictions
    
    def predict_dataframe(self, df, text_column='preprocessed_text', output_path=None):
        """
        Faz predições para um DataFrame.
        
        Args:
            df: DataFrame com textos
            text_column: Nome da coluna com textos
            output_path: Caminho para salvar resultados (opcional)
            
        Returns:
            DataFrame com predições
        """
        print(f"Fazendo predições para {len(df)} textos...")
        
        predictions, probabilities = self.predict_batch(
            df[text_column].values,
            return_probabilities=True
        )
        
        df_result = df.copy()
        df_result['predicted_label'] = predictions
        df_result['predicted_class'] = df_result['predicted_label'].map({
            0: 'True News',
            1: 'Fake News'
        })
        df_result['prob_true_news'] = [p['true_news'] for p in probabilities]
        df_result['prob_fake_news'] = [p['fake_news'] for p in probabilities]
        
        if output_path:
            df_result.to_parquet(output_path, index=False)
            print(f"Resultados salvos em: {output_path}")
        
        return df_result


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Fazer predições com modelo RoBERTa treinado'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Diretório contendo o modelo treinado'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Texto para classificar'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='Arquivo parquet com textos para classificar'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='preprocessed_text',
        help='Nome da coluna com textos'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Arquivo para salvar resultados'
    )
    
    args = parser.parse_args()
    
    # Inicializar preditor
    predictor = RoBERTaPredictor(args.model_dir)
    
    if args.text:
        # Predição para texto único
        pred, probs = predictor.predict(args.text, return_probabilities=True)
        
        print(f"\n{'='*60}")
        print("RESULTADO DA PREDIÇÃO")
        print(f"{'='*60}")
        print(f"Texto: {args.text[:200]}...")
        print(f"\nPredição: {'FAKE NEWS' if pred == 1 else 'TRUE NEWS'}")
        print(f"\nProbabilidades:")
        print(f"  True News: {probs['true_news']:.4f} ({probs['true_news']*100:.2f}%)")
        print(f"  Fake News: {probs['fake_news']:.4f} ({probs['fake_news']*100:.2f}%)")
    
    elif args.input_file:
        # Predição para arquivo
        df = pd.read_parquet(args.input_file)
        df_result = predictor.predict_dataframe(
            df, 
            text_column=args.text_column,
            output_path=args.output_file
        )
        
        print(f"\n{'='*60}")
        print("RESUMO DAS PREDIÇÕES")
        print(f"{'='*60}")
        print(f"Total de textos: {len(df_result)}")
        print(f"\nDistribuição das predições:")
        print(df_result['predicted_class'].value_counts())
        print(f"\nPorcentagem:")
        print(df_result['predicted_class'].value_counts(normalize=True) * 100)
        
        if 'is_fake' in df_result.columns:
            # Se temos labels verdadeiros, calcular acurácia
            df_result['correct'] = (
                df_result['predicted_label'] == df_result['is_fake'].astype(int)
            )
            accuracy = df_result['correct'].mean()
            print(f"\nAcurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    else:
        print("Por favor, forneça --text ou --input-file")
        parser.print_help()


if __name__ == '__main__':
    main()

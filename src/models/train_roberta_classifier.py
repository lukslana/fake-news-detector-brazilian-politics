"""
Script para treinamento de modelo RoBERTa (BERTimbau) para detecção de fake news.

Este script implementa um classificador de deep learning usando o modelo BERTimbau
pré-treinado em português para detectar notícias falsas no dataset FakeBr News.

Autor: Lucas Lana
Data: 2026-01-29
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    f1_score,
    precision_score,
    recall_score
)
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuração de visualização
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class NewsDataset(Dataset):
    """Dataset PyTorch para notícias."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RoBERTaTrainer:
    """Classe para treinar e avaliar modelo RoBERTa."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)}')
        
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.history = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': []
        }
        
    def load_data(self, data_path):
        """Carrega dados do arquivo parquet."""
        print(f"\n{'='*60}")
        print("CARREGANDO DADOS")
        print(f"{'='*60}")
        
        df = pd.read_parquet(data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"\nDistribuição de classes:")
        print(df['is_fake'].value_counts())
        print(f"\nPorcentagem:")
        print(df['is_fake'].value_counts(normalize=True) * 100)
        
        # Verificar dados faltantes
        print(f"\nValores nulos em preprocessed_text: {df['preprocessed_text'].isnull().sum()}")
        print(f"Valores vazios em preprocessed_text: {(df['preprocessed_text'] == '').sum()}")
        
        # Remover linhas com texto vazio ou nulo
        df_clean = df[df['preprocessed_text'].notna() & (df['preprocessed_text'] != '')].copy()
        print(f"Dataset após limpeza: {df_clean.shape}")
        
        return df_clean
    
    def prepare_data(self, df):
        """Prepara dados para treinamento."""
        print(f"\n{'='*60}")
        print("PREPARANDO DADOS")
        print(f"{'='*60}")
        
        # Split estratificado mantendo os pares juntos
        unique_sequences = df['sequence'].unique()
        train_sequences, test_sequences = train_test_split(
            unique_sequences, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Criar máscaras para train e test
        train_mask = df['sequence'].isin(train_sequences)
        test_mask = df['sequence'].isin(test_sequences)
        
        X_train = df[train_mask]['preprocessed_text'].values
        X_test = df[test_mask]['preprocessed_text'].values
        y_train = df[train_mask]['is_fake'].values.astype(int)
        y_test = df[test_mask]['is_fake'].values.astype(int)
        
        print(f"Tamanho do conjunto de treino: {len(X_train)}")
        print(f"Tamanho do conjunto de teste: {len(X_test)}")
        print(f"\nDistribuição no treino - Fake: {y_train.sum()}, True: {len(y_train) - y_train.sum()}")
        print(f"Distribuição no teste - Fake: {y_test.sum()}, True: {len(y_test) - y_test.sum()}")
        
        return X_train, X_test, y_train, y_test
    
    def setup_model(self):
        """Configura modelo, tokenizer e otimizador."""
        print(f"\n{'='*60}")
        print("CONFIGURANDO MODELO")
        print(f"{'='*60}")
        
        # Carregar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        print(f"Tokenizer carregado: {self.config['model_name']}")
        
        # Carregar modelo
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=2
        )
        self.model = self.model.to(self.device)
        
        print(f"Modelo carregado: {self.config['model_name']}")
        print(f"Total de parâmetros: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def create_data_loaders(self, X_train, X_test, y_train, y_test):
        """Cria DataLoaders para treinamento e teste."""
        print(f"\n{'='*60}")
        print("CRIANDO DATALOADERS")
        print(f"{'='*60}")
        
        train_dataset = NewsDataset(
            X_train, y_train, 
            self.tokenizer, 
            self.config['max_length']
        )
        test_dataset = NewsDataset(
            X_test, y_test, 
            self.tokenizer, 
            self.config['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size']
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        return train_loader, test_loader
    
    def setup_optimizer(self, train_loader):
        """Configura otimizador e scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        total_steps = len(train_loader) * self.config['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        print(f"Total de steps de treinamento: {total_steps}")
    
    def train_epoch(self, data_loader):
        """Treina o modelo por uma época."""
        self.model.train()
        losses = []
        correct_predictions = 0
        
        progress_bar = tqdm(data_loader, desc='Training')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)
    
    def eval_model(self, data_loader):
        """Avalia o modelo."""
        self.model.eval()
        losses = []
        correct_predictions = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return (
            correct_predictions.double() / len(data_loader.dataset), 
            np.mean(losses), 
            predictions, 
            true_labels
        )
    
    def train(self, train_loader, test_loader, save_dir):
        """Loop principal de treinamento."""
        print(f"\n{'='*60}")
        print("INICIANDO TREINAMENTO")
        print(f"{'='*60}")
        
        best_accuracy = 0
        
        for epoch in range(self.config['epochs']):
            print(f'\nEpoch {epoch + 1}/{self.config["epochs"]}')
            print('-' * 60)
            
            train_acc, train_loss = self.train_epoch(train_loader)
            print(f'Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}')
            
            val_acc, val_loss, _, _ = self.eval_model(test_loader)
            print(f'Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}')
            
            self.history['train_acc'].append(train_acc.item())
            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(val_acc.item())
            self.history['val_loss'].append(val_loss)
            
            if val_acc > best_accuracy:
                model_path = os.path.join(save_dir, 'roberta_best_model.bin')
                torch.save(self.model.state_dict(), model_path)
                best_accuracy = val_acc
                print(f'✓ Best model saved with accuracy: {best_accuracy:.4f}')
        
        return best_accuracy
    
    def plot_training_history(self, save_dir):
        """Plota histórico de treinamento."""
        print(f"\n{'='*60}")
        print("GERANDO GRÁFICOS")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_title('Perda por Época', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Perda')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', marker='s')
        axes[1].set_title('Acurácia por Época', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Acurácia')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {plot_path}")
        plt.close()
    
    def evaluate_final(self, test_loader, save_dir):
        """Avaliação final do modelo."""
        print(f"\n{'='*60}")
        print("AVALIAÇÃO FINAL")
        print(f"{'='*60}")
        
        # Carregar melhor modelo
        model_path = os.path.join(save_dir, 'roberta_best_model.bin')
        self.model.load_state_dict(torch.load(model_path))
        
        # Avaliar no conjunto de teste
        test_acc, test_loss, y_pred, y_true = self.eval_model(test_loader)
        
        print("\n=== RESULTADOS FINAIS ===")
        print(f"Acurácia no teste: {test_acc:.4f}")
        print(f"F1-Score no teste: {f1_score(y_true, y_pred):.4f}")
        print(f"Precision no teste: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall no teste: {recall_score(y_true, y_pred):.4f}")
        
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(
            y_true, y_pred, 
            target_names=['True News', 'Fake News'],
            digits=4
        ))
        
        # Salvar métricas
        metrics = {
            'accuracy': float(test_acc),
            'f1_score': float(f1_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'test_loss': float(test_loss),
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nMétricas salvas em: {metrics_path}")
        
        return y_true, y_pred, metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_dir):
        """Plota matriz de confusão."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['True News', 'Fake News'],
            yticklabels=['True News', 'Fake News'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Matriz de Confusão - RoBERTa', fontsize=14, fontweight='bold')
        plt.ylabel('Valor Real')
        plt.xlabel('Predição')
        plt.tight_layout()
        
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusão salva em: {cm_path}")
        plt.close()
        
        # Calcular métricas da matriz de confusão
        tn, fp, fn, tp = cm.ravel()
        print(f"\n=== MÉTRICAS DA MATRIZ DE CONFUSÃO ===")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")
        print(f"\nPrecisão: {tp/(tp+fp):.4f}")
        print(f"Recall: {tp/(tp+fn):.4f}")
    
    def save_training_history(self, save_dir):
        """Salva histórico de treinamento."""
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Histórico de treinamento salvo em: {history_path}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Treinar modelo RoBERTa para detecção de fake news'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/fakebr_news.parquet',
        help='Caminho para o arquivo de dados'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='neuralmind/bert-base-portuguese-cased',
        help='Nome do modelo pré-treinado'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Comprimento máximo das sequências'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Tamanho do batch'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Número de épocas'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Taxa de aprendizado'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporção do conjunto de teste'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed para reprodutibilidade'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models',
        help='Diretório para salvar modelos e resultados'
    )
    
    args = parser.parse_args()
    
    # Configuração
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'test_size': args.test_size,
        'random_state': args.random_state
    }
    
    # Criar diretório de salvamento
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Salvar configuração
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuração salva em: {config_path}")
    
    # Inicializar trainer
    trainer = RoBERTaTrainer(config)
    
    # Carregar dados
    df = trainer.load_data(args.data_path)
    
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
    best_accuracy = trainer.train(train_loader, test_loader, args.save_dir)
    
    # Plotar histórico
    trainer.plot_training_history(args.save_dir)
    
    # Salvar histórico
    trainer.save_training_history(args.save_dir)
    
    # Avaliação final
    y_true, y_pred, metrics = trainer.evaluate_final(test_loader, args.save_dir)
    
    # Plotar matriz de confusão
    trainer.plot_confusion_matrix(y_true, y_pred, args.save_dir)
    
    print(f"\n{'='*60}")
    print("TREINAMENTO CONCLUÍDO!")
    print(f"{'='*60}")
    print(f"Melhor acurácia: {best_accuracy:.4f}")
    print(f"Resultados salvos em: {args.save_dir}")


if __name__ == '__main__':
    main()

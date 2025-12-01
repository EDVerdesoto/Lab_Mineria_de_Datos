"""
Script principal de entrenamiento.

Uso:
    # Entrenamiento binario (Safe vs Vulnerable)
    python train.py --csv ../../data/dataset.csv --mode binary --epochs 5
    
    # Entrenamiento multi-clase (Safe + CWEs + Other)
    python train.py --csv ../../data/dataset.csv --mode multiclass --epochs 10
    
    # Con límite de muestras (para testing)
    python train.py --csv ../../data/dataset.csv --max-samples 10000
"""

import argparse
import sys
import os

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training import train_from_csv
from config import EPOCHS, BATCH_SIZE


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo de detección de vulnerabilidades')
    
    parser.add_argument(
        '--csv', 
        type=str, 
        required=True,
        help='Ruta al archivo CSV con el dataset'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='binary',
        choices=['binary', 'multiclass'],
        help='Modo de clasificación: binary (Safe/Vulnerable) o multiclass (CWEs)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Número de épocas (default: {EPOCHS})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Tamaño de batch (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Máximo número de muestras (para testing rápido)'
    )
    
    args = parser.parse_args()
    
    # Verificar que el archivo existe
    if not os.path.exists(args.csv):
        print(f"[ERROR] Archivo no encontrado: {args.csv}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"     ENTRENAMIENTO DE MODELO DE VULNERABILIDADES")
    print(f"{'='*60}")
    print(f"Dataset: {args.csv}")
    print(f"Modo: {args.mode}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print(f"{'='*60}\n")
    
    # Entrenar
    trainer, history = train_from_csv(
        csv_path=args.csv,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    print("\n[INFO] Entrenamiento completado.")
    print(f"[INFO] Mejor modelo guardado en: checkpoints/best_model.pt")


if __name__ == '__main__':
    main()

"""
Script de inferencia para analizar código.

Uso:
    # Analizar un archivo
    python predict.py --file vulnerable_code.py
    
    # Analizar código directo
    python predict.py --code "query = f'SELECT * FROM users WHERE id={user_id}'"
    
    # Analizar múltiples archivos
    python predict.py --dir ./src --ext py
    
    # Generar reporte JSON
    python predict.py --file code.py --format json
"""

import argparse
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import VulnerabilityPredictor
from config import CHECKPOINT_DIR


def main():
    parser = argparse.ArgumentParser(description='Analizar código en busca de vulnerabilidades')
    
    parser.add_argument(
        '--model',
        type=str,
        default=os.path.join(CHECKPOINT_DIR, 'best_model.pt'),
        help='Ruta al modelo entrenado'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Archivo de código a analizar'
    )
    parser.add_argument(
        '--code',
        type=str,
        help='Código directo a analizar (string)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Directorio con archivos a analizar'
    )
    parser.add_argument(
        '--ext',
        type=str,
        default='py',
        help='Extensión de archivos a buscar en directorio (default: py)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default=None,
        help='Lenguaje de programación (se infiere automáticamente si no se especifica)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='text',
        choices=['text', 'json'],
        help='Formato del reporte (default: text)'
    )
    
    args = parser.parse_args()
    
    # Verificar que se proporcionó al menos una entrada
    if not args.file and not args.code and not args.dir:
        print("[ERROR] Debe especificar --file, --code o --dir")
        parser.print_help()
        sys.exit(1)
    
    # Verificar que el modelo existe
    if not os.path.exists(args.model):
        print(f"[ERROR] Modelo no encontrado: {args.model}")
        print("[INFO] Primero entrene el modelo con: python train.py --csv data/dataset.csv")
        sys.exit(1)
    
    # Cargar predictor
    print(f"[INFO] Cargando modelo: {args.model}")
    predictor = VulnerabilityPredictor(args.model)
    
    # Analizar
    if args.code:
        # Código directo
        result = predictor.predict(args.code, args.language)
        report = predictor.generate_report(result, args.format)
        print(report)
    
    elif args.file:
        # Archivo único
        if not os.path.exists(args.file):
            print(f"[ERROR] Archivo no encontrado: {args.file}")
            sys.exit(1)
        
        result = predictor.analyze_file(args.file, args.language)
        report = predictor.generate_report(result, args.format)
        print(report)
    
    elif args.dir:
        # Directorio
        if not os.path.isdir(args.dir):
            print(f"[ERROR] Directorio no encontrado: {args.dir}")
            sys.exit(1)
        
        pattern = os.path.join(args.dir, f'**/*.{args.ext}')
        files = glob.glob(pattern, recursive=True)
        
        if not files:
            print(f"[WARN] No se encontraron archivos .{args.ext} en {args.dir}")
            sys.exit(0)
        
        print(f"[INFO] Analizando {len(files)} archivos...")
        
        vulnerable_files = []
        
        for file_path in files:
            try:
                result = predictor.analyze_file(file_path)
                
                if result['is_vulnerable']:
                    vulnerable_files.append(result)
                    print(f"[WARN]  {file_path}: {result['prediction']} ({result['confidence']:.1%})")
                else:
                    print(f"[INFO]  {file_path}: Safe")
            except Exception as e:
                print(f"[ERROR]  {file_path}: Error - {e}")
        
        # Resumen
        print(f"\n{'='*60}")
        print(f"     RESUMEN")
        print(f"{'='*60}")
        print(f"Total archivos: {len(files)}")
        print(f"Archivos vulnerables: {len(vulnerable_files)}")
        print(f"Archivos seguros: {len(files) - len(vulnerable_files)}")
        
        if vulnerable_files:
            print(f"\nArchivos vulnerables detectados:")
            for result in vulnerable_files:
                print(f"  - {result['file_path']}: {result['prediction']}")


if __name__ == '__main__':
    main()

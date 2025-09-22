#!/usr/bin/env python3
"""
Script para ejecutar el procesador de señales
"""

import sys
import os
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from core.ml.signals.signal_processor import create_signal_processor, process_signals_realtime, process_signals_batch

def main():
    parser = argparse.ArgumentParser(description="Ejecutar Procesador de Señales")
    parser.add_argument("--mode", choices=["realtime", "batch"], default="realtime",
                       help="Modo de procesamiento")
    parser.add_argument("--symbol", help="Símbolo específico a procesar")
    parser.add_argument("--timeframe", help="Timeframe específico a procesar")
    parser.add_argument("--lookback", type=int, default=5,
                       help="Minutos de lookback para tiempo real")
    parser.add_argument("--start-time", help="Hora de inicio para modo batch (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end-time", help="Hora de fin para modo batch (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--config", default="config/signals/signal_processing.yaml",
                       help="Ruta del archivo de configuración")
    parser.add_argument("--continuous", action="store_true",
                       help="Ejecutar continuamente (solo para modo realtime)")
    parser.add_argument("--interval", type=int, default=30,
                       help="Intervalo en segundos para ejecución continua")
    
    args = parser.parse_args()
    
    print("🚀 Iniciando Procesador de Señales")
    print("=" * 50)
    print(f"Modo: {args.mode}")
    print(f"Configuración: {args.config}")
    
    if args.mode == "realtime":
        if args.continuous:
            print(f"Ejecutando continuamente cada {args.interval} segundos...")
            print("Presiona Ctrl+C para detener")
            
            import time
            try:
                while True:
                    print(f"\n[{datetime.now()}] Procesando señales...")
                    stats = process_signals_realtime(args.config)
                    print(f"Resultado: {stats}")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n🛑 Deteniendo procesador...")
        else:
            print("Procesando señales una vez...")
            stats = process_signals_realtime(args.config)
            print(f"Resultado: {stats}")
    
    elif args.mode == "batch":
        if not args.start_time or not args.end_time:
            print("❌ Error: Modo batch requiere --start-time y --end-time")
            return
        
        try:
            start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")
            
            # Convertir a UTC si no tienen timezone
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            
            print(f"Procesando desde {start_time} hasta {end_time}")
            
            symbols = [args.symbol] if args.symbol else None
            stats = process_signals_batch(start_time, end_time, symbols, args.config)
            print(f"Resultado: {stats}")
            
        except ValueError as e:
            print(f"❌ Error en formato de fecha: {e}")
            print("Formato esperado: YYYY-MM-DD HH:MM:SS")

if __name__ == "__main__":
    main()

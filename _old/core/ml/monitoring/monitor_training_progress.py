#!/usr/bin/env python3
"""
Monitor de progreso de entrenamiento optimizado
"""

import os
import time
import pickle
import psutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path

def get_memory_usage():
    """Obtener uso de memoria en GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_system_memory():
    """Obtener información de memoria del sistema"""
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / (1024**3),
        'available': memory.available / (1024**3),
        'used': memory.used / (1024**3),
        'percent': memory.percent
    }

def check_checkpoint_status(symbol: str, timeframe: str, horizon: int):
    """Verificar estado del checkpoint"""
    checkpoint_path = f"logs/checkpoint_{symbol}_{timeframe}_H{horizon}.pkl"
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return {
            'exists': True,
            'completed_folds': len(checkpoint.get('completed_folds', [])),
            'best_auc': checkpoint.get('best_auc', 0),
            'last_update': os.path.getmtime(checkpoint_path),
            'fold_results': checkpoint.get('fold_results', [])
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}

def monitor_training_progress(symbol: str, timeframe: str, horizon: int, 
                            refresh_interval: int = 30):
    """Monitorear progreso del entrenamiento en tiempo real"""
    
    print(f"🔍 MONITOREANDO ENTRENAMIENTO: {symbol}-{timeframe} (H={horizon})")
    print("=" * 60)
    print(f"⏰ Actualización cada {refresh_interval} segundos")
    print("Presiona Ctrl+C para salir")
    print()
    
    try:
        while True:
            # Limpiar pantalla
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"🔍 MONITOREANDO ENTRENAMIENTO: {symbol}-{timeframe} (H={horizon})")
            print("=" * 60)
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Estado del checkpoint
            checkpoint_status = check_checkpoint_status(symbol, timeframe, horizon)
            
            if checkpoint_status is None:
                print("❌ No se encontró checkpoint - entrenamiento no iniciado")
            elif 'error' in checkpoint_status:
                print(f"❌ Error en checkpoint: {checkpoint_status['error']}")
            else:
                print("📊 ESTADO DEL ENTRENAMIENTO:")
                print(f"  Folds completados: {checkpoint_status['completed_folds']}")
                print(f"  Mejor AUC: {checkpoint_status['best_auc']:.4f}")
                
                last_update = datetime.fromtimestamp(checkpoint_status['last_update'])
                print(f"  Última actualización: {last_update.strftime('%H:%M:%S')}")
                
                # Mostrar métricas por fold
                fold_results = checkpoint_status['fold_results']
                if fold_results:
                    print("\n📈 MÉTRICAS POR FOLD:")
                    for i, result in enumerate(fold_results):
                        metrics = result['metrics']
                        print(f"  Fold {i+1}: AUC={metrics['auc']:.4f}, "
                              f"Brier={metrics['brier']:.4f}, "
                              f"Acc={metrics['acc']:.4f}")
            
            # Estado de memoria
            print("\n💾 ESTADO DE MEMORIA:")
            system_memory = get_system_memory()
            process_memory = get_memory_usage()
            
            print(f"  Sistema: {system_memory['used']:.1f}GB / {system_memory['total']:.1f}GB "
                  f"({system_memory['percent']:.1f}%)")
            print(f"  Proceso: {process_memory:.2f}GB")
            
            # Estado de archivos de log
            log_files = [
                "logs/train_direction.log",
                "logs/train_direction_optimized.log"
            ]
            
            print("\n📝 ARCHIVOS DE LOG:")
            for log_file in log_files:
                if os.path.exists(log_file):
                    size = os.path.getsize(log_file) / 1024  # KB
                    mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
                    print(f"  ✅ {log_file}: {size:.1f}KB (modificado: {mtime.strftime('%H:%M:%S')})")
                else:
                    print(f"  ❌ {log_file}: No existe")
            
            # Verificar si el entrenamiento está activo
            print("\n🔄 ESTADO DEL PROCESO:")
            training_active = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'train_direction' in cmdline and symbol in cmdline:
                        training_active = True
                        print(f"  ✅ Entrenamiento activo (PID: {proc.info['pid']})")
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not training_active:
                print("  ❌ Entrenamiento no detectado")
            
            print(f"\n⏳ Próxima actualización en {refresh_interval} segundos...")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoreo detenido por el usuario")

def show_training_summary(symbol: str, timeframe: str, horizon: int):
    """Mostrar resumen del entrenamiento"""
    
    print(f"📊 RESUMEN DE ENTRENAMIENTO: {symbol}-{timeframe} (H={horizon})")
    print("=" * 60)
    
    checkpoint_status = check_checkpoint_status(symbol, timeframe, horizon)
    
    if checkpoint_status is None:
        print("❌ No se encontró checkpoint")
        return
    
    if 'error' in checkpoint_status:
        print(f"❌ Error: {checkpoint_status['error']}")
        return
    
    print(f"✅ Folds completados: {checkpoint_status['completed_folds']}")
    print(f"🏆 Mejor AUC: {checkpoint_status['best_auc']:.4f}")
    
    fold_results = checkpoint_status['fold_results']
    if fold_results:
        print("\n📈 MÉTRICAS DETALLADAS:")
        print("Fold | AUC     | Brier   | Acc     | Tiempo")
        print("-" * 45)
        
        for i, result in enumerate(fold_results):
            metrics = result['metrics']
            print(f"{i+1:4d} | {metrics['auc']:.4f} | {metrics['brier']:.4f} | "
                  f"{metrics['acc']:.4f} | {metrics['train_time']:.1f}s")
        
        # Calcular promedios
        avg_auc = sum(r['metrics']['auc'] for r in fold_results) / len(fold_results)
        avg_brier = sum(r['metrics']['brier'] for r in fold_results) / len(fold_results)
        avg_acc = sum(r['metrics']['acc'] for r in fold_results) / len(fold_results)
        total_time = sum(r['metrics']['train_time'] for r in fold_results)
        
        print("-" * 45)
        print(f"Prom | {avg_auc:.4f} | {avg_brier:.4f} | {avg_acc:.4f} | {total_time:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Monitor de entrenamiento optimizado")
    parser.add_argument("--symbol", required=True, help="Símbolo a monitorear")
    parser.add_argument("--tf", required=True, help="Timeframe a monitorear")
    parser.add_argument("--horizon", type=int, default=1, help="Horizonte")
    parser.add_argument("--refresh", type=int, default=30, help="Intervalo de actualización en segundos")
    parser.add_argument("--summary", action="store_true", help="Mostrar solo resumen")
    
    args = parser.parse_args()
    
    if args.summary:
        show_training_summary(args.symbol, args.tf, args.horizon)
    else:
        monitor_training_progress(args.symbol, args.tf, args.horizon, args.refresh)

if __name__ == "__main__":
    main()

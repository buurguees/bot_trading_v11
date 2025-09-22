#!/usr/bin/env python3
"""
Script para verificar la estructura del proyecto reorganizado
"""

import os
import sys

def check_structure():
    print("🔍 VERIFICANDO ESTRUCTURA DEL PROYECTO")
    print("=" * 50)
    
    # Estructura esperada
    expected_structure = {
        "core/ml/training/_runs/": [
            "start_training.bat",
            "start_training_no_backfill.bat", 
            "start_training_optimized.bat",
            "README.md"
        ],
        "core/ml/monitoring/": [
            "check_duplicates.py",
            "check_recent_activity.py",
            "check_pnl_changes.py",
            "check_promotion_candidates.py",
            "monitor_emergency.py",
            "monitor_historical_backtests.py",
            "README.md"
        ],
        "core/ml/backtests/_scripts/": [
            # Scripts de backtesting (si los hay)
        ],
        "scripts/setup/": [
            "setup_full_data.py",
            "verify_data_setup.py"
        ],
        "scripts/maintenance/": [
            "fix_prediction_duplicates.py",
            "emergency_fix.py"
        ],
        "scripts/monitoring/": [
            # Duplicados de monitoreo (opcional)
        ]
    }
    
    # Scripts principales en raíz
    root_scripts = [
        "start_training.bat",
        "monitor_system.bat", 
        "maintenance.bat",
        "ESTRUCTURA_PROYECTO.md"
    ]
    
    all_good = True
    
    # Verificar estructura de directorios
    print("📁 Verificando directorios...")
    for dir_path in expected_structure.keys():
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - FALTANTE")
            all_good = False
    
    # Verificar archivos en directorios
    print("\n📄 Verificando archivos en directorios...")
    for dir_path, files in expected_structure.items():
        if os.path.exists(dir_path):
            for file in files:
                file_path = os.path.join(dir_path, file)
                if os.path.exists(file_path):
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ❌ {file_path} - FALTANTE")
                    all_good = False
    
    # Verificar scripts principales
    print("\n🚀 Verificando scripts principales...")
    for script in root_scripts:
        if os.path.exists(script):
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} - FALTANTE")
            all_good = False
    
    # Verificar archivos duplicados en raíz
    print("\n🧹 Verificando limpieza de raíz...")
    problematic_files = [
        "check_duplicates.py",
        "check_recent_activity.py", 
        "check_pnl_changes.py",
        "check_promotion_candidates.py",
        "monitor_emergency.py",
        "monitor_historical_backtests.py",
        "fix_prediction_duplicates.py",
        "emergency_fix.py",
        "setup_full_data.py",
        "verify_data_setup.py",
        "start_training_no_backfill.bat",
        "start_training_optimized.bat"
    ]
    
    for file in problematic_files:
        if os.path.exists(file):
            print(f"   ⚠️  {file} - DUPLICADO EN RAÍZ (debería estar en subdirectorio)")
            all_good = False
        else:
            print(f"   ✅ {file} - Correctamente ubicado")
    
    # Resumen
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 ¡ESTRUCTURA PERFECTA!")
        print("   Todos los archivos están en su lugar correcto.")
        print("   El proyecto está bien organizado.")
    else:
        print("⚠️  ESTRUCTURA INCOMPLETA")
        print("   Algunos archivos faltan o están mal ubicados.")
        print("   Revisa los errores anteriores.")
    
    return all_good

def show_usage():
    print("\n📋 CÓMO USAR LA NUEVA ESTRUCTURA:")
    print("-" * 40)
    print("🚀 Entrenamiento:")
    print("   start_training.bat")
    print("   core/ml/training/_runs/start_training_optimized.bat")
    print()
    print("📊 Monitoreo:")
    print("   monitor_system.bat")
    print("   core/ml/monitoring/monitor_emergency.py")
    print()
    print("🔧 Mantenimiento:")
    print("   maintenance.bat")
    print("   scripts/maintenance/fix_prediction_duplicates.py")
    print()
    print("⚙️  Configuración:")
    print("   scripts/setup/setup_full_data.py")
    print("   scripts/setup/verify_data_setup.py")

if __name__ == "__main__":
    success = check_structure()
    show_usage()
    
    if not success:
        sys.exit(1)
    else:
        print("\n✅ Verificación completada exitosamente!")

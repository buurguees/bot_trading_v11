#!/usr/bin/env python3
"""
Script de prueba rápida para verificar la configuración del proyecto
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Prueba las importaciones principales"""
    print("🔍 Probando importaciones...")
    
    try:
        import ccxt
        print("✅ CCXT importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando CCXT: {e}")
        return False
    
    try:
        import sqlalchemy
        print("✅ SQLAlchemy importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando SQLAlchemy: {e}")
        return False
    
    try:
        import psycopg2
        print("✅ psycopg2 importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando psycopg2: {e}")
        return False
    
    try:
        import talib
        print("✅ TA-Lib importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando TA-Lib: {e}")
        return False
    
    try:
        import telegram
        print("✅ python-telegram-bot importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando python-telegram-bot: {e}")
        return False
    
    try:
        import stable_baselines3
        print("✅ stable-baselines3 importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando stable-baselines3: {e}")
        return False
    
    return True

def test_config_files():
    """Verifica que los archivos de configuración existan"""
    print("\n🔍 Verificando archivos de configuración...")
    
    config_files = [
        "config/env.example",
        "config/trading/symbols.yaml",
        "config/trading/risk.yaml",
        "config/ml/training.yaml",
        "config/ml/rewards.yaml",
        "config/system/paths.yaml",
        "config/system/logging.yaml",
        "requirements.txt",
        ".gitignore"
    ]
    
    all_exist = True
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NO ENCONTRADO")
            all_exist = False
    
    return all_exist

def test_directory_structure():
    """Verifica la estructura de directorios"""
    print("\n🔍 Verificando estructura de directorios...")
    
    directories = [
        "agents",
        "config",
        "config/trading",
        "config/ml", 
        "config/system",
        "core",
        "core/data",
        "core/ml",
        "core/trading",
        "core/control",
        "data",
        "db",
        "db/migrations",
        "db/migrations/versions",
        "scripts",
        "scripts/initialization",
        "scripts/trading",
        "scripts/ml",
        "scripts/reporting",
        "tests",
        "docs"
    ]
    
    all_exist = True
    for dir_path in directories:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - NO ENCONTRADO")
            all_exist = False
    
    return all_exist

def test_scripts():
    """Verifica que los scripts principales existan"""
    print("\n🔍 Verificando scripts principales...")
    
    scripts = [
        "scripts/initialization/init_db.py",
        "scripts/initialization/init_db.sql",
        "scripts/initialization/verify_db.py",
        "scripts/initialization/README.md"
    ]
    
    all_exist = True
    for script_path in scripts:
        if Path(script_path).exists():
            print(f"✅ {script_path}")
        else:
            print(f"❌ {script_path} - NO ENCONTRADO")
            all_exist = False
    
    return all_exist

def main():
    """Función principal"""
    print("🚀 Bot Trading v11 - Verificación de Configuración")
    print("=" * 50)
    
    tests = [
        ("Importaciones de librerías", test_imports),
        ("Archivos de configuración", test_config_files),
        ("Estructura de directorios", test_directory_structure),
        ("Scripts principales", test_scripts)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if not test_func():
            all_passed = False
            print(f"❌ {test_name} - FALLÓ")
        else:
            print(f"✅ {test_name} - EXITOSO")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ¡Todas las verificaciones pasaron exitosamente!")
        print("🚀 El proyecto está listo para usar")
        print("\n📝 Próximos pasos:")
        print("1. Configurar PostgreSQL")
        print("2. Ejecutar: python scripts/initialization/init_db.py")
        print("3. Ejecutar: python scripts/initialization/verify_db.py")
        print("4. Configurar credenciales en config/.env")
    else:
        print("💥 Algunas verificaciones fallaron")
        print("🔧 Revisa los errores anteriores y corrige los problemas")
        sys.exit(1)

if __name__ == "__main__":
    main()

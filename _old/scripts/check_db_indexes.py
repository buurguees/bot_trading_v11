#!/usr/bin/env python3
"""
Script para verificar los índices creados en la base de datos
"""

import os
import sys
from sqlalchemy import create_engine, text

def create_connection():
    """Crear conexión a la base de datos"""
    db_url = "postgresql+psycopg2://trading_user:160501@192.168.10.109:5432/trading_db"
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine

def check_indexes():
    """Verificar índices existentes"""
    engine = create_connection()
    
    with engine.connect() as conn:
        # Verificar índices BRIN
        print("=" * 80)
        print("ÍNDICES BRIN CREADOS")
        print("=" * 80)
        
        result = conn.execute(text("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes 
            WHERE schemaname = 'trading' 
                AND indexname LIKE '%brin%'
            ORDER BY tablename, indexname
        """))
        
        for row in result:
            print(f"{row[0]}.{row[1]}: {row[2]}")
            print(f"  {row[3]}")
            print()
        
        # Verificar índices compuestos
        print("=" * 80)
        print("ÍNDICES COMPUESTOS CREADOS")
        print("=" * 80)
        
        result = conn.execute(text("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes 
            WHERE schemaname = 'trading' 
                AND (indexname LIKE '%symbol_tf%' OR indexname LIKE '%symbol_bar%')
            ORDER BY tablename, indexname
        """))
        
        for row in result:
            print(f"{row[0]}.{row[1]}: {row[2]}")
            print(f"  {row[3]}")
            print()
        
        # Estadísticas de índices
        print("=" * 80)
        print("ESTADÍSTICAS DE ÍNDICES")
        print("=" * 80)
        
        result = conn.execute(text("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                pg_size_pretty(pg_relation_size(schemaname||'.'||indexname)) as size
            FROM pg_indexes 
            WHERE schemaname = 'trading'
            ORDER BY pg_relation_size(schemaname||'.'||indexname) DESC
        """))
        
        for row in result:
            print(f"{row[0]}.{row[1]}.{row[2]}: {row[3]}")
        
        # Verificar tablas duplicadas
        print("\n" + "=" * 80)
        print("TABLAS DUPLICADAS (MAYÚSCULAS/MINÚSCULAS)")
        print("=" * 80)
        
        result = conn.execute(text("""
            SELECT 
                LOWER(tablename) as tablename,
                COUNT(*) as count
            FROM pg_tables 
            WHERE schemaname = 'trading'
            GROUP BY LOWER(tablename)
            HAVING COUNT(*) > 1
            ORDER BY LOWER(tablename)
        """))
        
        for row in result:
            print(f"Tabla duplicada: {row[0]} ({row[1]} versiones)")

if __name__ == "__main__":
    print("VERIFICACIÓN DE ÍNDICES DE BASE DE DATOS")
    print("=" * 80)
    print()
    
    try:
        check_indexes()
        print("\n" + "=" * 80)
        print("VERIFICACIÓN COMPLETADA")
        print("=" * 80)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

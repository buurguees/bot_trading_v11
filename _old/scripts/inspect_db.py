#!/usr/bin/env python3
"""
Script para inspeccionar la estructura de la base de datos
Muestra tablas, columnas, índices y estadísticas
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(dotenv_path="config/.env")

# Configuración de conexión
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://trading_user:160501@192.168.10.109:5432/trading_db')

def inspect_database():
    """Inspecciona la estructura completa de la base de datos"""
    
    print("🔍 INSPECCIÓN DE BASE DE DATOS - BOT TRADING V11")
    print("=" * 60)
    
    try:
engine = create_engine(DB_URL)

    with engine.connect() as conn:
            # 1. Listar todas las tablas en el esquema trading
            print("\n📊 TABLAS EN EL ESQUEMA 'trading':")
            print("-" * 40)
            
            tables_query = """
            SELECT 
                table_name,
                pg_size_pretty(pg_total_relation_size('trading.' || table_name)) as size
            FROM information_schema.tables 
            WHERE table_schema = 'trading'
            ORDER BY pg_total_relation_size('trading.' || table_name) DESC
            """
            
            result = conn.execute(text(tables_query))
            tables = result.fetchall()
            
            if not tables:
                print("❌ No se encontraron tablas en el esquema 'trading'")
                return
            
            for table in tables:
                print(f"  📋 {table[0]} ({table[1]})")
            
            # 2. Detalles de cada tabla
            for table_name, _ in tables:
                print(f"\n🔍 DETALLES DE LA TABLA: {table_name}")
                print("=" * 50)
                
                # Columnas de la tabla
                columns_query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns 
                WHERE table_schema = 'trading' AND table_name = :table_name
                ORDER BY ordinal_position
                """
                
                result = conn.execute(text(columns_query), {"table_name": table_name})
                columns = result.fetchall()
                
                print(f"📝 COLUMNAS ({len(columns)} total):")
                print("-" * 30)
                for col in columns:
                    col_name, data_type, nullable, default, max_length = col
                    length_info = f"({max_length})" if max_length else ""
                    nullable_info = "NULL" if nullable == "YES" else "NOT NULL"
                    default_info = f" DEFAULT {default}" if default else ""
                    print(f"  • {col_name}: {data_type}{length_info} {nullable_info}{default_info}")
                
                # Índices de la tabla
                indexes_query = """
                SELECT 
                    indexname,
                    indexdef
                FROM pg_indexes 
                WHERE schemaname = 'trading' AND tablename = :table_name
                ORDER BY indexname
                """
                
                result = conn.execute(text(indexes_query), {"table_name": table_name})
                indexes = result.fetchall()
                
                if indexes:
                    print(f"\n🔗 ÍNDICES ({len(indexes)} total):")
                    print("-" * 30)
                    for idx in indexes:
                        idx_name, idx_def = idx
                        print(f"  • {idx_name}")
                        print(f"    {idx_def}")
                
                # Estadísticas de la tabla
                stats_query = """
                SELECT 
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables 
                WHERE schemaname = 'trading' AND relname = :table_name
                """
                
                result = conn.execute(text(stats_query), {"table_name": table_name})
                stats = result.fetchone()
                
                if stats:
                    print(f"\n📈 ESTADÍSTICAS:")
                    print("-" * 30)
                    print(f"  • Tuplas vivas: {stats[3]:,}")
                    print(f"  • Tuplas muertas: {stats[4]:,}")
                    print(f"  • Inserts: {stats[0]:,}")
                    print(f"  • Updates: {stats[1]:,}")
                    print(f"  • Deletes: {stats[2]:,}")
                    if stats[5]:
                        print(f"  • Último VACUUM: {stats[5]}")
                    if stats[7]:
                        print(f"  • Último ANALYZE: {stats[7]}")
                
                # Muestra de datos (primeras 3 filas)
                sample_query = f"SELECT * FROM trading.{table_name} LIMIT 3"
                try:
                    result = conn.execute(text(sample_query))
                    sample_data = result.fetchall()
                    
                    if sample_data:
                        print(f"\n📄 MUESTRA DE DATOS (3 filas):")
                        print("-" * 30)
                        for i, row in enumerate(sample_data, 1):
                            print(f"  Fila {i}: {dict(row._mapping)}")
                    else:
                        print(f"\n📄 MUESTRA DE DATOS: Tabla vacía")
                        
                except Exception as e:
                    print(f"\n📄 MUESTRA DE DATOS: Error al obtener datos - {e}")
                
                print("\n" + "=" * 60)
            
            # 3. Resumen general
            print("\n📊 RESUMEN GENERAL:")
            print("-" * 30)
            
            total_tables = len(tables)
            total_size_query = """
            SELECT pg_size_pretty(SUM(pg_total_relation_size('trading.' || table_name)))
            FROM information_schema.tables
            WHERE table_schema = 'trading'
            """
            
            result = conn.execute(text(total_size_query))
            total_size = result.scalar()
            
            print(f"  • Total de tablas: {total_tables}")
            print(f"  • Tamaño total: {total_size}")
            
            # Verificar conexión
            conn.execute(text("SELECT 1"))
            print(f"  • Estado de conexión: ✅ Activa")
            
    except Exception as e:
        print(f"❌ Error al conectar con la base de datos: {e}")
        print(f"   URL de conexión: {DB_URL}")
        return False
    
    return True

def show_specific_table(table_name):
    """Muestra detalles de una tabla específica"""
    
    print(f"🔍 DETALLES DE LA TABLA: {table_name}")
    print("=" * 50)
    
    try:
        engine = create_engine(DB_URL)
        
    with engine.connect() as conn:
            # Verificar que la tabla existe
            check_query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'trading' AND table_name = :table_name
            )
            """
            
            result = conn.execute(text(check_query), {"table_name": table_name})
            exists = result.scalar()
            
            if not exists:
                print(f"❌ La tabla 'trading.{table_name}' no existe")
                return False
            
            # Obtener columnas
            columns_query = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = 'trading' AND table_name = :table_name
            ORDER BY ordinal_position
            """
            
            result = conn.execute(text(columns_query), {"table_name": table_name})
            columns = result.fetchall()
            
            print(f"📝 COLUMNAS ({len(columns)} total):")
            for col in columns:
                col_name, data_type, nullable, default = col
                nullable_info = "NULL" if nullable == "YES" else "NOT NULL"
                default_info = f" DEFAULT {default}" if default else ""
                print(f"  • {col_name}: {data_type} {nullable_info}{default_info}")
            
            # Contar registros
            count_query = f"SELECT COUNT(*) FROM trading.{table_name}"
            result = conn.execute(text(count_query))
            count = result.scalar()
            
            print(f"\n📊 Total de registros: {count:,}")
            
            # Muestra de datos
            if count > 0:
                sample_query = f"SELECT * FROM trading.{table_name} LIMIT 5"
                result = conn.execute(text(sample_query))
                sample_data = result.fetchall()
                
                print(f"\n📄 MUESTRA DE DATOS (5 filas):")
                for i, row in enumerate(sample_data, 1):
                    print(f"  Fila {i}: {dict(row._mapping)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Mostrar tabla específica
        table_name = sys.argv[1]
        show_specific_table(table_name)
    else:
        # Mostrar todas las tablas
        inspect_database()
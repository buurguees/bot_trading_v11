#!/usr/bin/env python3
"""
Script para limpiar duplicados en agentpreds y prevenir futuros duplicados
"""

from core.data.database import get_engine
from sqlalchemy import text

def fix_prediction_duplicates():
    print("🧹 LIMPIANDO DUPLICADOS EN PREDICCIONES")
    print("=" * 50)
    
    eng = get_engine()
    
    try:
        with eng.connect() as conn:
            # 1. Verificar duplicados actuales
            print("1. Verificando duplicados actuales...")
            
            check_query = """
            SELECT 
                symbol,
                timeframe,
                timestamp,
                COUNT(*) as duplicados
            FROM trading.agentpreds
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY symbol, timeframe, timestamp
            HAVING COUNT(*) > 1
            ORDER BY duplicados DESC;
            """
            
            result = conn.execute(text(check_query))
            duplicates = result.fetchall()
            
            if duplicates:
                print(f"   🔴 Encontrados {len(duplicates)} timestamps con duplicados")
                total_duplicates = sum(row[3] - 1 for row in duplicates)  # -1 porque uno es válido
                print(f"   📊 Total de registros duplicados a eliminar: {total_duplicates}")
            else:
                print("   ✅ No hay duplicados")
                return
            
            # 2. Crear tabla temporal con registros únicos
            print("2. Creando tabla temporal con registros únicos...")
            
            create_temp_query = """
            CREATE TEMP TABLE temp_unique_predictions AS
            SELECT DISTINCT ON (symbol, timeframe, timestamp)
                id,
                agent_version_id,
                symbol,
                timeframe,
                timestamp,
                horizon,
                payload,
                created_at
            FROM trading.agentpreds
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            ORDER BY symbol, timeframe, timestamp, created_at DESC;
            """
            
            conn.execute(text(create_temp_query))
            print("   ✅ Tabla temporal creada")
            
            # 3. Contar registros únicos
            count_query = """
            SELECT COUNT(*) FROM temp_unique_predictions;
            """
            result = conn.execute(text(count_query))
            unique_count = result.scalar()
            print(f"   📊 Registros únicos: {unique_count}")
            
            # 4. Eliminar duplicados de la tabla original
            print("3. Eliminando duplicados de la tabla original...")
            
            delete_query = """
            DELETE FROM trading.agentpreds
            WHERE id NOT IN (
                SELECT id FROM temp_unique_predictions
            )
            AND created_at >= NOW() - INTERVAL '24 hours';
            """
            
            result = conn.execute(text(delete_query))
            deleted_count = result.rowcount
            print(f"   ✅ Eliminados {deleted_count} registros duplicados")
            
            # 5. Verificar que no quedan duplicados
            print("4. Verificando que no quedan duplicados...")
            
            verify_query = """
            SELECT 
                symbol,
                timeframe,
                timestamp,
                COUNT(*) as duplicados
            FROM trading.agentpreds
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY symbol, timeframe, timestamp
            HAVING COUNT(*) > 1;
            """
            
            result = conn.execute(text(verify_query))
            remaining_duplicates = result.fetchall()
            
            if remaining_duplicates:
                print(f"   🔴 Aún quedan {len(remaining_duplicates)} timestamps con duplicados")
            else:
                print("   ✅ No quedan duplicados")
            
            # 6. Crear índice único para prevenir futuros duplicados
            print("5. Creando índice único para prevenir futuros duplicados...")
            
            try:
                create_index_query = """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_agentpreds_unique_prediction
                ON trading.agentpreds (symbol, timeframe, timestamp);
                """
                conn.execute(text(create_index_query))
                print("   ✅ Índice único creado")
            except Exception as e:
                print(f"   ⚠️ Error creando índice (puede que ya exista): {e}")
            
            # 7. Limpiar planes de trading duplicados
            print("6. Limpiando planes de trading duplicados...")
            
            clean_plans_query = """
            DELETE FROM trading.tradeplans
            WHERE id NOT IN (
                SELECT DISTINCT ON (symbol, timeframe, bar_ts, side)
                    id
                FROM trading.tradeplans
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY symbol, timeframe, bar_ts, side, created_at DESC
            )
            AND created_at >= NOW() - INTERVAL '24 hours';
            """
            
            result = conn.execute(text(clean_plans_query))
            deleted_plans = result.rowcount
            print(f"   ✅ Eliminados {deleted_plans} planes duplicados")
            
            # 8. Commit de todos los cambios
            conn.commit()
            print("   ✅ Cambios guardados en la base de datos")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ LIMPIEZA DE DUPLICADOS COMPLETADA")
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. Reinicia el entrenamiento")
    print("2. Monitorea que no se generen más duplicados")
    print("3. El índice único prevendrá futuros duplicados")
    
    return True

if __name__ == "__main__":
    fix_prediction_duplicates()

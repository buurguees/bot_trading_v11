#!/usr/bin/env python3
"""
Monitor de Entrenamiento ML
==========================
Script para monitorear la evolución del entrenamiento mediante consultas SQL.

Uso:
    python monitor_training.py [--hours 24] [--task direction|regime|smc]
"""

import argparse
from datetime import datetime, timedelta
from core.data.database import MarketDB
from sqlalchemy import text

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def monitor_predictions(db, hours=24, task=None):
    """Monitorear predicciones de agentes"""
    print_header("PREDICCIONES DE AGENTES")
    
    where_clause = f"WHERE created_at >= NOW() - INTERVAL '{hours} hours'"
    if task:
        where_clause += f" AND task = '{task}'"
    
    # Predicciones por tarea
    with db.engine.begin() as conn:
        result = conn.execute(text(f"""
            SELECT 
                task,
                COUNT(*) as total_predicciones,
                AVG(pred_conf) as confianza_promedio,
                MIN(pred_conf) as confianza_min,
                MAX(pred_conf) as confianza_max,
                MAX(created_at) as ultima_prediccion
            FROM ml.agent_preds 
            {where_clause}
            GROUP BY task
            ORDER BY task
        """))
        
        for row in result:
            print(f"📊 {row[0].upper()}:")
            print(f"   Total: {row[1]} predicciones")
            print(f"   Confianza: {row[2]:.3f} (min: {row[3]:.3f}, max: {row[4]:.3f})")
            print(f"   Última: {row[5]}")
            print()

def monitor_trade_plans(db, hours=24):
    """Monitorear trade plans generados"""
    print_header("TRADE PLANS GENERADOS")
    
    with db.engine.begin() as conn:
        # Por estado
        result = conn.execute(text(f"""
            SELECT 
                status,
                COUNT(*) as cantidad,
                AVG(confidence) as confianza_promedio,
                MAX(created_at) as ultimo_plan
            FROM trading.trade_plans 
            WHERE created_at >= NOW() - INTERVAL '{hours} hours'
            GROUP BY status
            ORDER BY cantidad DESC
        """))
        
        print("📈 Por Estado:")
        for row in result:
            print(f"   {row[0]}: {row[1]} planes, conf: {row[2]:.3f}, último: {row[3]}")
        
        # Por símbolo
        result = conn.execute(text(f"""
            SELECT 
                symbol,
                COUNT(*) as planes,
                AVG(confidence) as confianza_promedio
            FROM trading.trade_plans 
            WHERE created_at >= NOW() - INTERVAL '{hours} hours'
            GROUP BY symbol
            ORDER BY planes DESC
            LIMIT 10
        """))
        
        print("\n📊 Por Símbolo (Top 10):")
        for row in result:
            print(f"   {row[0]}: {row[1]} planes, conf: {row[2]:.3f}")

def monitor_strategies(db):
    """Monitorear estrategias minadas"""
    print_header("ESTRATEGIAS MINADAS")
    
    with db.engine.begin() as conn:
        # Por estado
        result = conn.execute(text("""
            SELECT 
                status,
                COUNT(*) as cantidad,
                MAX(updated_at) as ultima_actualizacion
            FROM ml.strategies 
            GROUP BY status
            ORDER BY cantidad DESC
        """))
        
        for row in result:
            print(f"🎯 {row[0]}: {row[1]} estrategias, última: {row[2]}")
        
        # Estrategias recientes
        result = conn.execute(text("""
            SELECT 
                strategy_id,
                symbol,
                timeframe,
                status,
                created_at
            FROM ml.strategies 
            ORDER BY created_at DESC
            LIMIT 5
        """))
        
        print("\n🆕 Estrategias Recientes:")
        for row in result:
            strategy_id = str(row[0])[:8] + "..."
            print(f"   {strategy_id} | {row[1]} {row[2]} | {row[3]} | {row[4]}")

def monitor_agents(db):
    """Monitorear agentes registrados"""
    print_header("AGENTES REGISTRADOS")
    
    with db.engine.begin() as conn:
        # Resumen general
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_agentes,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as activos,
                COUNT(CASE WHEN promoted_at IS NOT NULL THEN 1 END) as promovidos,
                MAX(created_at) as ultimo_registro
            FROM ml.agents
        """))
        
        row = result.fetchone()
        print(f"🤖 Total agentes: {row[0]}")
        print(f"✅ Activos: {row[1]}")
        print(f"⭐ Promovidos: {row[2]}")
        print(f"🕒 Último registro: {row[3]}")
        
        # Por tarea
        result = conn.execute(text("""
            SELECT 
                task,
                COUNT(*) as cantidad,
                AVG(CAST(metrics->>'accuracy' AS FLOAT)) as accuracy_promedio
            FROM ml.agents
            WHERE metrics->>'accuracy' IS NOT NULL
            GROUP BY task
            ORDER BY task
        """))
        
        print("\n📊 Por Tarea:")
        for row in result:
            accuracy = row[2] if row[2] else 0.0
            print(f"   {row[0]}: {row[1]} agentes, accuracy: {accuracy:.3f}")

def monitor_performance_trends(db, hours=24):
    """Monitorear tendencias de rendimiento"""
    print_header("TENDENCIAS DE RENDIMIENTO")
    
    with db.engine.begin() as conn:
        # Predicciones por hora
        result = conn.execute(text(f"""
            SELECT 
                DATE_TRUNC('hour', created_at) as hora,
                task,
                COUNT(*) as predicciones,
                AVG(pred_conf) as confianza_promedio
            FROM ml.agent_preds 
            WHERE created_at >= NOW() - INTERVAL '{hours} hours'
            GROUP BY DATE_TRUNC('hour', created_at), task
            ORDER BY hora DESC, task
            LIMIT 15
        """))
        
        print("📈 Predicciones por Hora:")
        for row in result:
            print(f"   {row[0]} | {row[1]}: {row[2]} pred, conf: {row[3]:.3f}")

def monitor_data_quality(db):
    """Monitorear calidad de datos"""
    print_header("CALIDAD DE DATOS")
    
    with db.engine.begin() as conn:
        # Features por símbolo
        result = conn.execute(text("""
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as total_features,
                MAX(ts) as ultimo_feature,
                MIN(ts) as primer_feature
            FROM market.features
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """))
        
        print("📊 Features por Símbolo/Timeframe:")
        for row in result:
            print(f"   {row[0]} {row[1]}: {row[2]} features ({row[4]} → {row[3]})")
        
        # Datos históricos
        result = conn.execute(text("""
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as total_velas,
                MAX(ts) as ultima_vela,
                MIN(ts) as primera_vela
            FROM market.historical_data
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """))
        
        print("\n📈 Datos Históricos:")
        for row in result:
            print(f"   {row[0]} {row[1]}: {row[2]} velas ({row[4]} → {row[3]})")

def main():
    parser = argparse.ArgumentParser(description='Monitor de Entrenamiento ML')
    parser.add_argument('--hours', type=int, default=24, help='Horas hacia atrás para analizar')
    parser.add_argument('--task', choices=['direction', 'regime', 'smc'], help='Filtrar por tarea específica')
    parser.add_argument('--section', choices=['predictions', 'plans', 'strategies', 'agents', 'trends', 'quality', 'all'], 
                       default='all', help='Sección específica a mostrar')
    
    args = parser.parse_args()
    
    print(f"🔍 MONITOR DE ENTRENAMIENTO ML - Últimas {args.hours}h")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    db = MarketDB()
    
    try:
        if args.section in ['predictions', 'all']:
            monitor_predictions(db, args.hours, args.task)
        
        if args.section in ['plans', 'all']:
            monitor_trade_plans(db, args.hours)
        
        if args.section in ['strategies', 'all']:
            monitor_strategies(db)
        
        if args.section in ['agents', 'all']:
            monitor_agents(db)
        
        if args.section in ['trends', 'all']:
            monitor_performance_trends(db, args.hours)
        
        if args.section in ['quality', 'all']:
            monitor_data_quality(db)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print(f"\n✅ Monitor completado - {datetime.now().strftime('%H:%M:%S')}")
    return 0

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Script para verificar qué modelos podrían promoverse con los nuevos umbrales
"""

import pandas as pd
from core.data.database import get_engine
from sqlalchemy import text

def check_promotion_candidates():
    print("🎯 VERIFICANDO CANDIDATOS A PROMOCIÓN")
    print("=" * 50)
    
    eng = get_engine()
    
    # Configuración de umbrales ajustados
    thresholds = {
        "BTCUSDT": {"min_auc": 0.52, "max_brier": 0.26, "min_acc": 0.50, "ratio": 2.0},
        "ETHUSDT": {"min_auc": 0.52, "max_brier": 0.26, "min_acc": 0.50, "ratio": 2.0},
        "ADAUSDT": {"min_auc": 0.50, "max_brier": 0.30, "min_acc": 0.50, "ratio": 1.1},
        "SOLUSDT": {"min_auc": 0.50, "max_brier": 0.30, "min_acc": 0.50, "ratio": 1.1},
        "DOGEUSDT": {"min_auc": 0.50, "max_brier": 0.30, "min_acc": 0.50, "ratio": 1.1},
        "XRPUSDT": {"min_auc": 0.50, "max_brier": 0.30, "min_acc": 0.50, "ratio": 1.1}
    }
    
    query = """
    SELECT 
        (params->>'symbol') as symbol,
        (params->>'timeframe') as timeframe,
        id as version_id,
        created_at,
        COALESCE((metrics->>'auc')::float8, (params->'metrics'->>'auc')::float8) as auc,
        COALESCE((metrics->>'brier')::float8, (params->'metrics'->>'brier')::float8) as brier,
        COALESCE((metrics->>'acc')::float8, (params->'metrics'->>'acc')::float8) as accuracy,
        promoted
    FROM trading.agentversions
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    ORDER BY created_at DESC;
    """
    
    try:
        with eng.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if df.empty:
                print("❌ No hay modelos en las últimas 24 horas")
                return
            
            print(f"📊 Total de modelos en las últimas 24h: {len(df)}")
            print()
            
            # Analizar cada símbolo
            for symbol in thresholds.keys():
                symbol_data = df[df['symbol'] == symbol]
                if symbol_data.empty:
                    continue
                
                print(f"🔍 {symbol}:")
                print(f"   Objetivo: {thresholds[symbol]['ratio']}x")
                print(f"   Umbrales: AUC≥{thresholds[symbol]['min_auc']}, Brier≤{thresholds[symbol]['max_brier']}, Acc≥{thresholds[symbol]['min_acc']}")
                
                # Filtrar por timeframe
                for timeframe in ['1m', '5m']:
                    tf_data = symbol_data[symbol_data['timeframe'] == timeframe]
                    if tf_data.empty:
                        continue
                    
                    print(f"   {timeframe}:")
                    
                    # Verificar candidatos a promoción
                    candidates = tf_data[
                        (tf_data['auc'] >= thresholds[symbol]['min_auc']) &
                        (tf_data['brier'] <= thresholds[symbol]['max_brier']) &
                        (tf_data['accuracy'] >= thresholds[symbol]['min_acc'])
                    ]
                    
                    if not candidates.empty:
                        print(f"     ✅ {len(candidates)} candidatos a promoción:")
                        for _, row in candidates.iterrows():
                            status = "PROMOVIDO" if row['promoted'] else "PENDIENTE"
                            print(f"       - ID {row['version_id']}: AUC={row['auc']:.4f}, Brier={row['brier']:.4f}, Acc={row['accuracy']:.4f} [{status}]")
                    else:
                        print(f"     ❌ 0 candidatos a promoción")
                        
                        # Mostrar el mejor modelo
                        best = tf_data.loc[tf_data['auc'].idxmax()]
                        print(f"       Mejor: ID {best['version_id']}: AUC={best['auc']:.4f}, Brier={best['brier']:.4f}, Acc={best['accuracy']:.4f}")
                        
                        # Mostrar qué falta
                        missing = []
                        if best['auc'] < thresholds[symbol]['min_auc']:
                            missing.append(f"AUC (+{thresholds[symbol]['min_auc'] - best['auc']:.4f})")
                        if best['brier'] > thresholds[symbol]['max_brier']:
                            missing.append(f"Brier (-{best['brier'] - thresholds[symbol]['max_brier']:.4f})")
                        if best['accuracy'] < thresholds[symbol]['min_acc']:
                            missing.append(f"Acc (+{thresholds[symbol]['min_acc'] - best['accuracy']:.4f})")
                        
                        if missing:
                            print(f"       Falta: {', '.join(missing)}")
                
                print()
            
            # Resumen general
            print("📈 RESUMEN GENERAL:")
            total_candidates = 0
            total_promoted = 0
            
            for symbol in thresholds.keys():
                symbol_data = df[df['symbol'] == symbol]
                for timeframe in ['1m', '5m']:
                    tf_data = symbol_data[symbol_data['timeframe'] == timeframe]
                    if tf_data.empty:
                        continue
                    
                    candidates = tf_data[
                        (tf_data['auc'] >= thresholds[symbol]['min_auc']) &
                        (tf_data['brier'] <= thresholds[symbol]['max_brier']) &
                        (tf_data['accuracy'] >= thresholds[symbol]['min_acc'])
                    ]
                    
                    total_candidates += len(candidates)
                    total_promoted += len(candidates[candidates['promoted'] == True])
            
            print(f"   Total candidatos: {total_candidates}")
            print(f"   Total promovidos: {total_promoted}")
            print(f"   Pendientes: {total_candidates - total_promoted}")
            
            if total_candidates > 0:
                print(f"   Tasa de promoción: {(total_promoted / total_candidates * 100):.1f}%")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_promotion_candidates()

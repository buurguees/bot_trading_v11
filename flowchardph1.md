flowchart LR
  subgraph EXCHANGE["Mercado: Bitget REST/WS"]
    MWS["WebSocket"]
    MREST["REST"]
  end

  subgraph DB1["PostgreSQL (schemas)"]
    HD["market.historical_data"]
    FEAT["market.features"]
    PRED["ml.agent_preds"]
    TP["trading.trade_plans"]
    ORD["trading.orders"]
    FIL["trading.fills"]
    POS["trading.positions"]
    RISK["risk.risk_events"]
  end

  subgraph INGESTA["Ingesta de datos"]
    HDL["core/data/historical_downloader.py"]
    RTU["core/data/realtime_updater.py"]
  end

  subgraph FE["Feature Engineering"]
    FENG["core/ml/feature_engineer.py"]
  end

  subgraph HEADS["Agentes (cabezas)"]
    DIR["core/agents/agent_direction.py"]
    REG["core/agents/agent_regime.py"]
    SMC["core/agents/agent_smc.py"]
  end

  subgraph PLAN["Planificación y Riesgo"]
    PLNR["core/trading/planner.py"]
    RM["core/trading/risk_manager.py"]
  end

  subgraph OMS["OMS / Enrutador"]
    ROUT["core/trading/oms/router.py"]
    POSMOD["core/trading/oms/positions.py"]
  end

  subgraph GUI["Observabilidad"]
    GUI1["scripts/gui_training_monitor.py"]
    MON["monitor_training.py"]
  end

  MREST --> HDL
  MWS   --> RTU
  HDL   --> HD
  RTU   --> HD
  HD    --> FENG
  FENG  --> FEAT

  FEAT --> DIR
  FEAT --> REG
  FEAT --> SMC
  DIR  --> PRED
  REG  --> PRED
  SMC  --> PRED

  PRED --> PLNR
  PLNR --> TP
  TP   --> RM
  RM   --> TP
  RM   --> RISK

  TP   --> ROUT
  ROUT --> ORD
  ROUT --> FIL
  ROUT --> POS
  POS  --> POSMOD

  PRED --> GUI1
  TP   --> GUI1
  ORD  --> GUI1
  FIL  --> GUI1
  POS  --> GUI1
  RISK --> GUI1

  PRED --> MON
  TP   --> MON
  ORD  --> MON
  FIL  --> MON
  POS  --> MON
  RISK --> MON

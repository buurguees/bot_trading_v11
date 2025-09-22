flowchart TD
    %% =====================
    %% CAPAS DE FEATURES
    %% =====================
    subgraph Input_Features["Input & Features"]
        F1[market.historical_data] --> FE[core/ml/feature_engineer.py]
        FE --> F2[market.features]
    end

    %% =====================
    %% ENCODER / ATTENTION
    %% =====================
    subgraph Encoder["Encoder Temporal + Cross-TF Attention"]
        ENC[core/ml/encoders/multitf_encoder.py]
    end
    F2 --> ENC

    %% =====================
    %% CABEZAS NEURONALES
    %% =====================
    subgraph Heads["Neuronas (Heads)"]
        DIR[core/ml/models/direction.py]
        REG[core/ml/models/regime.py]
        SMC[core/ml/models/smc.py]
        EXEC[core/ml/policy/ppo_execution.py]
    end
    ENC --> DIR
    ENC --> REG
    ENC --> SMC
    ENC --> EXEC

    %% =====================
    %% AGENTES
    %% =====================
    subgraph Agents["Agentes Orquestadores"]
        ADIR[core/agents/agent_direction.py]
        AREG[core/agents/agent_regime.py]
        ASMC[core/agents/agent_smc.py]
        AEXE[core/agents/agent_execution.py]
    end
    DIR --> ADIR
    REG --> AREG
    SMC --> ASMC
    EXEC --> AEXE

    %% Predicciones a BD
    ADIR --> PRED[ml.agent_preds]
    AREG --> PRED
    ASMC --> PRED
    AEXE --> PRED

    %% =====================
    %% PLANNER + RISK
    %% =====================
    subgraph PlannerRisk["Planner & Risk"]
        PLAN[core/trading/planner.py]
        RISK[core/trading/risk_manager.py]
    end
    ADIR --> PLAN
    AREG --> PLAN
    ASMC --> PLAN
    AEXE --> PLAN

    PLAN --> TP[trading.trade_plans]
    PLAN --> RISK
    RISK --> RE[risk.risk_events]

    %% =====================
    %% OMS
    %% =====================
    subgraph OMS["Order Management System"]
        ROUTER[core/trading/oms/router.py]
        ORD[trading.orders]
        FILLS[trading.fills]
        POS[trading.positions]
    end
    RISK --> ROUTER
    ROUTER --> ORD
    ROUTER --> FILLS
    ROUTER --> POS

    %% =====================
    %% MONITORING & AUTOLEARN
    %% =====================
    subgraph Monitoring["Monitoring & AutoLearn"]
        AUTO[core/ml/autolearn.py]
        MET[monitoring.metrics]
        STRAT[ml.strategies_memory]
    end
    PRED --> AUTO
    POS --> AUTO
    AUTO --> STRAT
    AUTO --> MET

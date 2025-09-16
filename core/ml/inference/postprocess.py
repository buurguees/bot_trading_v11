def prob_to_side(prob_up: float, buy_th: float = 0.55, sell_th: float = 0.45) -> int:
    if prob_up >= buy_th:  return  +1
    if prob_up <= sell_th: return  -1
    return 0

def strength_from_prob(prob_up: float) -> float:
    return float(abs(prob_up - 0.5) * 2.0)  # 0..1

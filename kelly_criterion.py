def kelly_criterion(probability, bookmaker_odds):
    fraction = probability - ((1-probability)/(bookmaker_odds-1))
    return fraction
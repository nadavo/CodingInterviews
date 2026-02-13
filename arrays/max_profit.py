from typing import List


def maxProfitSingle(prices: List[int]) -> int:
    if len(prices) <= 1:
        return 0
    elif len(prices) == 2:
        return max(0, prices[1] - prices[0])

    min_price = prices[0]
    max_profit = 0
    for p in prices[1:]:
        min_price = min(p, min_price)
        max_profit = max(p - min_price, max_profit)

    return max_profit


def maxProfitMultiple(prices: List[int]) -> int:
    if len(prices) <= 1:
        return 0
    elif len(prices) == 2:
        return max(0, prices[1] - prices[0])

    max_profit = sum([max(0, prices[i] - prices[i - 1]) for i in range(1, len(prices))])

    return max_profit

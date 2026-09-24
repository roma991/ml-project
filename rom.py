
from decimal import Decimal, getcontext

getcontext().prec = 50

def solve():
    I0 = Decimal(7).ln() - Decimal(6).ln()
    I_direct = [I0]
    for n in range(1, 32):
        I_direct.append((Decimal(1) / Decimal(n)) - Decimal(6) * I_direct[-1])
    result_a = I_direct[31]

    I_reverse = [Decimal(0)] * 61
    for n in range(60, 31, -1):
        I_reverse[n - 1] = (Decimal(1) / (Decimal(6) * Decimal(n))) - (I_reverse[n] / Decimal(6))
    result_b = I_reverse[31]

    total_sum = Decimal(0)
    step = Decimal(1) / Decimal(1000)
    half = step / Decimal(2)
    for i in range(1000):
        x = Decimal(i) * step + half
        total_sum += (x ** 31) / (x + Decimal(6))
    result_c = total_sum * step
    print(result_a)
    print(result_b)
    print(result_c)
solve()
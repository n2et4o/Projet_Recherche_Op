def maximize_on_cycle(x, cycle):
    """
    Applique le déplacement de quantité sur le cycle.
    cycle = [ ((i,j), '+'), ((i,j), '-'), ... ]
    """
    theta = float("inf")

    # 1. Trouver le min sur les '-'
    for (i, j), sign in cycle:
        if sign == '-':
            theta = min(theta, x[i][j])

    # 2. Appliquer +θ ou -θ
    for (i, j), sign in cycle:
        if sign == '+':
            x[i][j] += theta
        else:
            x[i][j] -= theta

    return x

 

def compute_reduced_costs(cost, c_star):
    """
    Calcule la matrice des coûts marginaux :
        delta_ij = cost_ij - c_star_ij

    Paramètres
    ----------
    cost   : matrice des coûts réels
    c_star : matrice des coûts potentiels

    Retour
    ------
    delta : matrice des coûts marginaux
    """

    n = len(cost)
    m = len(cost[0])

    delta = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            delta[i][j] = cost[i][j] - c_star[i][j]

    return delta

 

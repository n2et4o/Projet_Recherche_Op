def compute_potential_costs(x, E):
    """
    Calcule la matrice des coûts potentiels :
        c*_ij = E(S_i) - E(C_j)

    Paramètres
    ----------
    x : matrice des quantités (pour récupérer n et m)
    E : dictionnaire des potentiels

    Retour
    ------
    c_star : matrice (n x m) des coûts potentiels
    """

    n = len(x)
    m = len(x[0])

    # matrice vide
    c_star = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            c_star[i][j] = E[f"S{i}"] - E[f"C{j}"]

    return c_star

 

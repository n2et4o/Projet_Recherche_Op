def find_entering_arc(x, delta):
    """
    Trouve la meilleure arête entrante à partir des coûts marginaux.

    Paramètres
    ----------
    x     : matrice des quantités (sert à déterminer les cases non basiques)
    delta : matrice des coûts marginaux Δ_ij

    Retour
    ------
    (i, j) si une arête entrante existe (Δ_ij le plus négatif)
    None   sinon -> optimalité atteinte
    """

    n = len(x)
    m = len(x[0])

    best = None
    best_val = 0  # on cherche le plus NEGATIF

    for i in range(n):
        for j in range(m):

            # On ne prend que les cases NON BASIQUES (= quantité 0)
            if x[i][j] != 0:
                continue

            # Si Δ_ij est plus négatif que le meilleur actuel
            if delta[i][j] < best_val:
                best_val = delta[i][j]
                best = (i, j)

    # Si aucune valeur négative → optimalité
    return best
 

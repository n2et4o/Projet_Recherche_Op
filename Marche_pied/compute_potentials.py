def compute_potentials(x, cost, basis):
    """
    Calcule les potentiels E(S_i) et E(C_j) à partir des cases basiques.

    Paramètres
    ----------
    x     : matrice des quantités
    cost  : matrice des coûts
    basis : matrice booléenne (True si basique)

    Retour
    ------
    E : dict des potentiels
        ex : { "S0": 0, "S1": 12, "C0": -5, ... }
    """

    n = len(x)
    m = len(x[0])

    # 1. Récupération des équations basiques
    equations = []   # liste de tuples (Si, Cj, cost_ij)

    for i in range(n):
        for j in range(m):
            if x[i][j] > 0 or basis[i][j] is True:
                equations.append(("S"+str(i), "C"+str(j), cost[i][j]))

    # 2. Déterminer le sommet le plus connecté
    count = {}

    for S, C, c in equations:
        count[S] = count.get(S, 0) + 1
        count[C] = count.get(C, 0) + 1

    # sommet à fixer
    root = max(count, key=count.get)

    # 3. Initialisation des potentiels
    E = {s: None for s in count}
    E[root] = 0  # celui qui apparaît le plus dans les équations

    # 4. Propagation
    changed = True
    while changed:
        changed = False

        for S, C, cij in equations:

            # afficher l'équation
            print(f"E({S}) - E({C}) = {cij}")

            if E[S] is not None and E[C] is None:
                E[C] = E[S] - cij
                print(f"  ⇒  E({C}) = E({S}) - {cij} = {E[C]}")
                changed = True

            elif E[C] is not None and E[S] is None:
                E[S] = E[C] + cij
                print(f"  ⇒  E({S}) = E({C}) + {cij} = {E[S]}")
                changed = True

    return E
 

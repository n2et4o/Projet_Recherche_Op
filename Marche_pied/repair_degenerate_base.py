def repair_degenerate_base(x, basis, cost, graph, visited):
    """
    Répare une base dégénérée lorsqu'elle n'est pas connexe.
    On ajoute une arête basique (i,j) de coût minimal reliant deux composantes.

    Paramètres
    ----------
    x      : matrice des quantités
    basis  : matrice booléenne de la base
    cost   : matrice des coûts
    graph  : graphe construit par build_graph
    visited: ensemble des sommets atteints (les autres sont problématiques)

    Retour
    ------
    basis modifié (matrice basique mise à jour)
    """

    n = len(x)
    m = len(x[0])

    # 1) Sommets problématiques
    all_nodes = set(graph.keys())
    problematic = all_nodes - visited

    # Séparation lignes / colonnes
    problematic_S = {int(s[1:]) for s in problematic if s.startswith("S")}
    problematic_C = {int(c[1:]) for c in problematic if c.startswith("C")}

    # Sommets corrects
    visited_S = {int(s[1:]) for s in visited if s.startswith("S")}
    visited_C = {int(c[1:]) for c in visited if c.startswith("C")}

    # 2) Recherche de la case (i,j) candidate
    best_i, best_j = None, None
    best_cost = float("inf")

    for i in range(n):
        for j in range(m):

            # doit être non basique
            if basis[i][j] is True:
                continue

            # doit être vide (quantité 0)
            if x[i][j] != 0:
                continue

            # doit relier une composante isolée à une composante visitée
            connects_components = (
                (i in problematic_S and j in visited_C) or
                (i in visited_S and j in problematic_C)
            )

            if not connects_components:
                continue

            # choisir le coût minimal
            if cost[i][j] < best_cost:
                best_cost = cost[i][j]
                best_i, best_j = i, j

    # 3) Ajouter l'arête basique choisie
    if best_i is not None:
        basis[best_i][best_j] = True

    return basis


 

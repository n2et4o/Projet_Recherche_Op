def build_graph(x, basis):
    """
    Construit le graphe biparti correspondant à la base du transport.

    Sommets :
        S0, S1, ..., S(n-1)
        C0, C1, ..., C(m-1)

    Arêtes :
        (Si, Cj) pour chaque case basique, c’est-à-dire :
        - basis[i][j] == True
          OU
        - x[i][j] > 0   (variable basique non dégénérée)

    Paramètres
    ----------
    x : matrice des quantités (n x m)
    basis : matrice booléenne basique/non-basique

    Retour
    ------
    graph : dictionnaire adjacency-list
            { "S0":[...], "S1":[...], ..., "C0":[...] ... }
    """

    n = len(x)
    m = len(x[0])

    # Initialisation des sommets
    graph = {f"S{i}": [] for i in range(n)}
    graph.update({f"C{j}": [] for j in range(m)})

    # Ajout des arêtes basiques
    for i in range(n):
        for j in range(m):
            if x[i][j] > 0 or basis[i][j] is True:
                s = f"S{i}"
                c = f"C{j}"
                graph[s].append(c)
                graph[c].append(s)

    # Affichage graphique via la fonction de ton camarade
    afficher_graphe(graph)

    return graph

from collections import deque

def is_connected(graph):
    """
    Teste si le graphe biparti est connexe en utilisant un BFS.

    Paramètres
    ----------
    graph : dict (adjacency list)
            { "S0":[...], "C0":[...], ... }

    Retour
    ------
    (bool, visited)
        - True  si tous les sommets sont atteints
        - False sinon + l'ensemble des sommets visités
    """

    if not graph:
        return True, set()   # graphe vide → connexe par définition

    # On prend n'importe quel sommet comme point de départ
    start = next(iter(graph))
    visited = set([start])
    queue = deque([start])

    # BFS
    while queue:
        node = queue.popleft()
        for neigh in graph[node]:
            if neigh not in visited:
                visited.add(neigh)
                queue.append(neigh)

    # Si on n'a pas visité tous les sommets → graphe non connexe
    is_connexe = (len(visited) == len(graph))

    return is_connexe, visited





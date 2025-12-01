
from collections import deque

def is_acyclic(graph):
    """
    Teste si le graphe contient un cycle via un parcours en largeur (BFS).

    Paramètre
    ---------
    graph : dict  (adjacency list)

    Retour
    ------
    True  : le graphe est acyclique
    False : un cycle a été détecté
    """

    visited = set()

    # Le graphe peut être composé de plusieurs composantes
    for start in graph:
        if start not in visited:
            queue = deque([(start, None)])  # (sommet, parent)

            while queue:
                node, parent = queue.popleft()

                if node in visited:
                    # revisite d'un sommet = cycle
                    return False

                visited.add(node)

                for neigh in graph[node]:
                    if neigh != parent:
                        queue.append((neigh, node))

    return True


 

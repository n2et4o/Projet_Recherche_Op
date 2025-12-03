import networkx as nx
import matplotlib.pyplot as plt

def dessiner_graphe_basis(basis):
    """
    Dessine le graphe biparti à partir de la matrice 'basis'.
    Une arête (Pi, Cj) est dessinée si basis[i][j] == True.
    """

    n = len(basis)
    m = len(basis[0])

    # --- Création du graphe ---
    G = nx.Graph()

    # Ajouter les sommets Pi (fournisseurs)
    sources = [f"P{i+1}" for i in range(n)]
    # Ajouter les sommets Cj (clients)
    destinations = [f"C{j+1}" for j in range(m)]

    for s in sources:
        G.add_node(s, bipartite=0)
    for c in destinations:
        G.add_node(c, bipartite=1)

    # --- Ajouter les arêtes basiques ---
    for i in range(n):
        for j in range(m):
            if basis[i][j] == True:
                G.add_edge(f"P{i+1}", f"C{j+1}")

    # --- Position des nœuds (disposition bipartie propre) ---
    pos = {}

    for i, s in enumerate(sources):
        pos[s] = (0, n - i)

    for j, c in enumerate(destinations):
        pos[c] = (4, m - j)

    # --- Dessin ---
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=12
    )

    plt.title("Graphe basique à partir de basis")
    plt.axis('off')
    plt.show()

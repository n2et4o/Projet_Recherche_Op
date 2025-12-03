from collections import deque
from math import inf
from copy import deepcopy

# ============================================================
#  Couleurs console (ANSI)
# ============================================================
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def color(text, *styles):
    """Retourne le texte entouré des styles ANSI donnés."""
    return "".join(styles) + str(text) + RESET


# ============================================================
#  Affichage du graphe (optionnel, via module externe)
# ============================================================
try:
    from fonctions_dessiner_graphe import afficher_graphe
except ImportError:
    def afficher_graphe(graph):
        print("\nAffichage texte du graphe biparti :")
        for node, neighs in graph.items():
            print(f"  {node} -> {neighs}")


# ============================================================
#  Affichages génériques
# ============================================================
def afficher_tableau(titre, mat):
    """Affichage simple sans labels."""
    print(f"\n=== {titre} ===")
    for ligne in mat:
        print(" ".join(f"{val:>4}" for val in ligne))


def afficher_matrice_labels(titre, mat, formatter=str):
    """
    Affiche une matrice n x m avec en-têtes S_i / C_j.
    Utilisé pour c*, Δ, etc.
    """
    n = len(mat)
    m = len(mat[0]) if n > 0 else 0

    print(f"\n=== {titre} ===")
    # En-tête colonnes
    header = "      " + " ".join(f"{'C'+str(j):>6}" for j in range(m))
    print(header)
    print("      " + "-" * (6 * m))

    for i in range(n):
        cellules = []
        for j in range(m):
            val = formatter(mat[i][j])
            cellules.append(f"{val:>6}")
        print(f"{'S'+str(i):>3} |" + "".join(cellules))


def afficher_matrice_transport(titre, mat, provisions, commandes, highlight=None):
    """
    Affiche une matrice de transport (coûts ou quantités)
    avec format type 2 :

           C0   C1   C2   C3 |  Prov
    --------------------------------
    S0     ..   ..   ..   .. |  60
    ...
    --------------------------------
          50   75   30   25

    highlight : (i,j) à mettre en évidence (cyan gras).
    """
    n = len(mat)
    m = len(mat[0]) if n > 0 else 0

    print(f"\n=== {titre} ===")

    # En-tête
    header = "      |" + "".join(f"{('C'+str(j)):>6}" for j in range(m)) + " |  Prov"
    print(header)
    print("      " + "-" * (6 * m + 11))

    # Lignes S_i
    for i in range(n):
        row_cells = []
        for j in range(m):
            val = mat[i][j]
            txt = f"{val:>6}"
            if highlight is not None and (i, j) == highlight:
                txt = color(txt, BOLD, CYAN)
            row_cells.append(txt)
        prov_txt = f"{provisions[i]:>5}"
        print(f"{'S'+str(i):>3} |" + "".join(row_cells) + " |" + prov_txt)

    # Séparateur
    print("      " + "-" * (6 * m + 11))

    # Ligne des commandes
    cmd_cells = "".join(f"{d:>6}" for d in commandes)
    print(" Cmd  |" + cmd_cells)


def afficher_couts(cost, provisions=None, commandes=None):
    """
    Affiche la matrice des coûts.
    Si provisions et commandes sont fournis, utilise le format transport.
    """
    if provisions is not None and commandes is not None:
        afficher_matrice_transport("Matrice des coûts (Cij)", cost, provisions, commandes)
    else:
        afficher_matrice_labels("Matrice des coûts (Cij)", cost, formatter=lambda v: v)


def afficher_quantites(x, provisions=None, commandes=None):
    """
    Affiche la matrice des quantités X.
    Si provisions et commandes sont fournis, utilise le format transport.
    """
    if provisions is not None and commandes is not None:
        afficher_matrice_transport("Matrice des quantités (Xij)", x, provisions, commandes)
    else:
        afficher_matrice_labels("Matrice des quantités (Xij)", x, formatter=lambda v: v)


def afficher_basis(basis):
    def fmt(v):
        return "B" if v else "."
    afficher_matrice_labels("Matrice BASIS (B = basique)", basis, formatter=fmt)


def afficher_c_star(c_star):
    afficher_matrice_labels("Coûts potentiels c* (E(Si) - E(Cj))", c_star, formatter=lambda v: v)


def afficher_delta(delta, basis, entering):
    """
    Affiche la matrice des coûts marginaux Δ avec :
    - toutes les valeurs négatives en rouge (hors base)
    - l'arête entrante en rouge gras
    """
    n = len(delta)
    m = len(delta[0])

    if entering is None:
        print("\nTous les coûts marginaux Δ_ij sont >= 0 sur les cases non basiques :")
        print("la base actuelle est OPTIMALE.")
    else:
        i_e, j_e = entering
        val_e = delta[i_e][j_e]
        print(
            "\n"
            f"Arête entrante choisie : ({i_e}, {j_e}) avec Δ_ij = "
            f"{color(val_e, BOLD, RED)} (le plus négatif parmi les cases non basiques)."
        )

    print("\n=== Coûts marginaux Δ = c - c* (Δ<0 en rouge, arête entrante en rouge gras) ===")
    print("      " + " ".join(f"{'C'+str(j):>6}" for j in range(m)))
    print("      " + "-" * (6 * m))

    for i in range(n):
        cellule_ligne = []
        for j in range(m):
            v = delta[i][j]
            txt = f"{v:>6}"
            if entering is not None and (i, j) == entering:
                txt = color(txt, BOLD, RED)
            elif v < 0 and not basis[i][j]:
                txt = color(txt, RED)
            cellule_ligne.append(txt)
        print(f"{'S'+str(i):>3} |" + "".join(cellule_ligne))


# ============================================================
#  Construction du graphe biparti
# ============================================================
def build_graph(x, basis):
    """
    Construit le graphe biparti correspondant à la base du transport.
    Sommets : S0..S(n-1), C0..C(m-1)
    Arêtes : (Si, Cj) si la case est basique (basis True ou x>0)
    """
    n = len(x)
    m = len(x[0])

    graph = {f"S{i}": [] for i in range(n)}
    graph.update({f"C{j}": [] for j in range(m)})

    for i in range(n):
        for j in range(m):
            if x[i][j] > 0 or basis[i][j] is True:
                s = f"S{i}"
                c = f"C{j}"
                graph[s].append(c)
                graph[c].append(s)

    afficher_graphe(graph)
    return graph


# ============================================================
#  Tests de connexité / acyclicité
# ============================================================
def is_connected(graph):
    """
    Teste si le graphe biparti est connexe via BFS.
    Retourne (bool, visited)
    """
    if not graph:
        return True, set()

    start = next(iter(graph))
    visited = {start}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neigh in graph[node]:
            if neigh not in visited:
                visited.add(neigh)
                queue.append(neigh)

    return len(visited) == len(graph), visited


def is_acyclic(graph):
    """
    Test simple de présence de cycle via BFS.
    Retourne True si acyclique, False si un cycle est détecté.
    """
    visited = set()

    for start in graph:
        if start not in visited:
            queue = deque([(start, None)])

            while queue:
                node, parent = queue.popleft()

                if node in visited:
                    # revisite = cycle
                    return False

                visited.add(node)

                for neigh in graph[node]:
                    if neigh != parent:
                        queue.append((neigh, node))

    return True


# ============================================================
#  Potentiels, coûts potentiels, coûts marginaux
# ============================================================
def compute_potentials(x, cost, basis):
    """
    Calcule les potentiels E(S_i) et E(C_j) à partir des cases basiques.
    Retourne un dict E : { "S0":0, "S1":..., "C0":..., ... }
    """
    n = len(x)
    m = len(x[0])

    equations = []
    for i in range(n):
        for j in range(m):
            if x[i][j] > 0 or basis[i][j] is True:
                equations.append((f"S{i}", f"C{j}", cost[i][j]))

    if not equations:
        raise ValueError("Aucune équation basique trouvée pour calculer les potentiels.")

    # nombre d'apparitions de chaque sommet
    count = {}
    for S, C, cij in equations:
        count[S] = count.get(S, 0) + 1
        count[C] = count.get(C, 0) + 1

    root = max(count, key=count.get)
    print(
        "\nOn choisit le sommet de référence "
        f"{color(root, BOLD, CYAN)} avec E({root}) = 0.\n"
    )

    E = {s: None for s in count}
    E[root] = 0

    changed = True
    while changed:
        changed = False
        for S, C, cij in equations:
            print(f"E({S}) - E({C}) = {cij}")

            if E[S] is not None and E[C] is None:
                E[C] = E[S] - cij
                print(f"  ⇒  E({C}) = E({S}) - {cij} = {E[C]}")
                changed = True
            elif E[C] is not None and E[S] is None:
                E[S] = E[C] + cij
                print(f"  ⇒  E({S}) = E({C}) + {cij} = {E[S]}")
                changed = True

    print("\nPotentiels finaux :")
    for s in sorted(E.keys()):
        mark = " (référence)" if s == root else ""
        print(f"  E({s}) = {E[s]}{mark}")
    print()

    return E


def compute_potential_costs(x, E):
    """
    c*_ij = E(S_i) - E(C_j)
    Retourne la matrice c_star (n x m)
    """
    n = len(x)
    m = len(x[0])
    c_star = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            c_star[i][j] = E[f"S{i}"] - E[f"C{j}"]

    return c_star


def compute_reduced_costs(cost, c_star):
    """
    delta_ij = cost_ij - c*_ij
    """
    n = len(cost)
    m = len(cost[0])
    delta = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            delta[i][j] = cost[i][j] - c_star[i][j]

    return delta


# ============================================================
#  Choix de l'arête entrante
# ============================================================
def find_entering_arc(x, basis, delta):
    """
    Trouve l'arête entrante (i,j) correspondant à Δ_ij le plus négatif
    parmi les cases NON BASIQUES.
    Retourne None si toutes les Δ_ij >= 0.
    """
    n = len(x)
    m = len(x[0])

    best = None
    best_val = 0  # on cherche le plus négatif

    for i in range(n):
        for j in range(m):
            # on ne prend que les NON BASIQUES
            if basis[i][j]:
                continue

            if delta[i][j] < best_val:
                best_val = delta[i][j]
                best = (i, j)

    return best


# ============================================================
#  Construction du cycle pour l'arête entrante
# ============================================================
def build_cycle_for_entering_arc(x, basis, entering):
    """
    Construit le cycle associé à l'arête entrante (i,j)
    et retourne la liste [ ((i,j), '+'), ((i,j), '-'), ... ].
    """
    i0, j0 = entering
    n = len(x)
    m = len(x[0])

    # base temporaire incluant l'arête entrante
    temp_basis = [row[:] for row in basis]
    temp_basis[i0][j0] = True

    # graphe biparti
    graph = {}
    for i in range(n):
        graph[f"S{i}"] = []
    for j in range(m):
        graph[f"C{j}"] = []

    for i in range(n):
        for j in range(m):
            if temp_basis[i][j] is True or x[i][j] > 0:
                graph[f"S{i}"].append(f"C{j}")
                graph[f"C{j}"].append(f"S{i}")

    start = f"S{i0}"
    target = f"C{j0}"

    queue = deque([start])
    parent = {start: None}

    while queue:
        node = queue.popleft()
        for neigh in graph[node]:
            # on interdit l'arête directe entrante
            if node == f"S{i0}" and neigh == f"C{j0}":
                continue
            if neigh not in parent:
                parent[neigh] = node
                queue.append(neigh)
                if neigh == target:
                    queue.clear()
                    break

    if target not in parent:
        raise ValueError("Impossible de construire un cycle pour l'arête entrante")

    # reconstruction du chemin
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    # conversion en positions (i,j)
    cycle_positions = []
    for k in range(len(path) - 1):
        a = path[k]
        b = path[k + 1]
        if a.startswith("S"):
            i = int(a[1:])
            j = int(b[1:])
        else:
            j = int(a[1:])
            i = int(b[1:])
        cycle_positions.append((i, j))

    # on ajoute l'arête entrante au début
    cycle_positions.insert(0, (i0, j0))

    cycle = []
    sign = '+'
    for pos in cycle_positions:
        cycle.append((pos, sign))
        sign = '-' if sign == '+' else '+'

    return cycle


# ============================================================
#  Maximisation sur le cycle
# ============================================================
def maximize_on_cycle(x, cycle):
    """
    Applique le déplacement de quantité sur le cycle.
    cycle = [ ((i,j), '+'), ((i,j), '-'), ... ]
    """
    theta = float("inf")

    for (i, j), sign in cycle:
        if sign == '-':
            theta = min(theta, x[i][j])

    print(f"\n> Maximisation sur le cycle avec θ = {theta}\n")
    for (i, j), sign in cycle:
        old = x[i][j]
        if sign == '+':
            x[i][j] += theta
            op_symb = '+'
            s_col = GREEN
        else:
            x[i][j] -= theta
            op_symb = '-'
            s_col = RED
        print(
            f"  x[{i},{j}] {color(op_symb, s_col)}= {theta} : {old} → {x[i][j]}"
        )

    return x


# ============================================================
#  Réparation d'une base dégénérée
# ============================================================
def repair_degenerate_base(x, basis, cost, graph, visited):
    """
    Répare une base dégénérée non connexe en ajoutant une case basique
    de coût minimal reliant deux composantes.
    Retourne (basis, added) avec added = (i,j) ou None.
    """
    n = len(x)
    m = len(x[0])

    all_nodes = set(graph.keys())
    problematic = all_nodes - visited

    problematic_S = {int(s[1:]) for s in problematic if s.startswith("S")}
    problematic_C = {int(c[1:]) for c in problematic if c.startswith("C")}

    visited_S = {int(s[1:]) for s in visited if s.startswith("S")}
    visited_C = {int(c[1:]) for c in visited if c.startswith("C")}

    isolated = [node for node in problematic if not graph[node]]
    if problematic:
        print(color("\nBase NON connexe.", YELLOW))
        print("Sommets non reliés à la base :", problematic)
        if isolated:
            print("Parmi eux, sommets totalement isolés :", isolated)

    best_i, best_j = None, None
    best_cost = float("inf")

    for i in range(n):
        for j in range(m):
            if basis[i][j] is True:
                continue
            if x[i][j] != 0:
                continue

            connects_components = (
                (i in problematic_S and j in visited_C) or
                (i in visited_S and j in problematic_C)
            )
            if not connects_components:
                continue

            if cost[i][j] < best_cost:
                best_cost = cost[i][j]
                best_i, best_j = i, j

    added = None
    if best_i is not None:
        print(
            "\n→ On cherche une case (i,j) de coût minimal reliant ces composantes."
        )
        print(
            f"→ On ajoute la case basique ({best_i}, {best_j}) "
            f"de coût {cost[best_i][best_j]} pour réparer la base."
        )
        basis[best_i][best_j] = True
        added = (best_i, best_j)

    return basis, added


# ============================================================
#  Méthode du marche-pied complète
# ============================================================
def marche_pied(x, basis, cost):
    """
    Méthode du marche-pied complète.
    Améliore la base jusqu'à atteindre l'optimalité.
    """
    iteration = 1

    while True:
        print("\n====================================")
        print(f"     ITERATION {iteration}")
        print("====================================")

        # Proposition actuelle
        afficher_quantites(x)
        afficher_basis(basis)

        # 1. Graphe de la base
        print("\n> Graphe de la base :")
        graph = build_graph(x, basis)

        # 2. Dégénérescence : connexité + acyclicité
        print("\n> Test de connexité...")
        connexe, visited = is_connected(graph)
        if connexe:
            print(color("Graphe connexe : toutes les lignes et colonnes sont reliées.", GREEN))
        else:
            print(color("Graphe NON connexe : certaines lignes/colonnes sont isolées.", YELLOW))

        print("> Test d'acyclicité...")
        acyclique = is_acyclic(graph)
        if acyclique:
            print(color("Graphe sans cycle parasite (acyclique).", GREEN))
        else:
            print(color("Cycle détecté dans la base (dégénérescence possible).", YELLOW))

        if not connexe or not acyclique:
            print(color("⚠️  Base dégénérée : tentative de réparation...", YELLOW))
            basis, added = repair_degenerate_base(x, basis, cost, graph, visited)
            if added is not None:
                print(
                    "→ La base est réparée en ajoutant la case "
                    f"{color(added, BOLD, CYAN)}."
                )
            afficher_basis(basis)
            graph = build_graph(x, basis)

        # 3. Potentiels
        print("\n> Calcul des potentiels :")
        E = compute_potentials(x, cost, basis)

        # 4. Coûts potentiels c*
        c_star = compute_potential_costs(x, E)
        afficher_c_star(c_star)

        # 5. Coûts marginaux Δ + arête entrante
        delta = compute_reduced_costs(cost, c_star)
        entering = find_entering_arc(x, basis, delta)
        afficher_delta(delta, basis, entering)

        if entering is None:
            print("\n✅ Solution optimale atteinte !")
            afficher_quantites(x)
            return x, basis

        print(f"\n> Arête entrante retenue : {color(entering, BOLD, RED)}")

        # 7. Cycle correspondant
        cycle = build_cycle_for_entering_arc(x, basis, entering)

        print("\nCycle (positions du cycle avec signes) :")
        for (i, j), sign in cycle:
            col = GREEN if sign == '+' else RED
            print(f"   ({i}, {j}) ({color(sign, col)})")

        # 8. Maximisation
        x = maximize_on_cycle(x, cycle)
        print("\nVoici les matrices mises à jour après déplacement sur le cycle :")
        afficher_quantites(x)

        # 9. Mise à jour de la base
        ei, ej = entering
        basis[ei][ej] = True
        afficher_basis(basis)

        iteration += 1


# ============================================================
#  Affichage Balas-Hammer détaillé
# ============================================================
def afficher_etat_balas(x, offre, demande,
                        penalites_lignes=None, penalites_colonnes=None,
                        i_choisie=None, j_choisie=None,
                        idx_pen_ligne=None, idx_pen_col=None,
                        delta_max=None, choisir_ligne=True):
    """
    Affiche l'état de la matrice X, des provisions / demandes
    et des pénalités pour Balas-Hammer.
    """
    n = len(x)
    m = len(x[0]) if n > 0 else 0

    print("\n--- État courant de la méthode Balas-Hammer ---")

    # En-tête
    header = "      |" + "".join(f"{('C'+str(j)):>6}" for j in range(m)) + " |  Prov | Δ_ligne"
    print(header)
    print("      " + "-" * (6 * m + 19))

    # Lignes
    for i in range(n):
        row_cells = []
        for j in range(m):
            val = x[i][j]
            txt = f"{val:>6}"
            if i_choisie is not None and j_choisie is not None and (i, j) == (i_choisie, j_choisie):
                txt = color(txt, BOLD, CYAN)
            row_cells.append(txt)

        prov_txt = f"{offre[i]:>5}"

        if penalites_lignes is not None:
            pl = penalites_lignes[i]
            if pl is None or pl < 0:
                pl_txt = "   -"
            else:
                pl_txt = f"{pl:>4}"
                if idx_pen_ligne is not None and i == idx_pen_ligne and delta_max is not None:
                    pl_txt = color(pl_txt, BOLD, RED)
        else:
            pl_txt = "   -"

        print(f"{'S'+str(i):>3} |" + "".join(row_cells) + " |" + prov_txt + " | " + pl_txt)

    # Séparateur
    print("      " + "-" * (6 * m + 19))

    # Ligne des commandes
    cmd_cells = "".join(f"{d:>6}" for d in demande)
    print(" Cmd  |" + cmd_cells + " |" + " " * 7)

    # Ligne des pénalités colonnes
    if penalites_colonnes is not None:
        line = "Δcol |"
        for j in range(m):
            pc = penalites_colonnes[j]
            if pc is None or pc < 0:
                pc_txt = "   - "
            else:
                pc_txt = f"{pc:>4} "
                if idx_pen_col is not None and j == idx_pen_col and delta_max is not None:
                    pc_txt = color(pc_txt, BOLD, RED)
            line += pc_txt
        print(line)

    # Phrase d'explication
    if delta_max is not None:
        if choisir_ligne and idx_pen_ligne is not None:
            print(
                f"\nLa pénalité maximale est Δ = {color(delta_max, BOLD, RED)} "
                f"sur la ligne S{idx_pen_ligne}."
            )
        elif not choisir_ligne and idx_pen_col is not None:
            print(
                f"\nLa pénalité maximale est Δ = {color(delta_max, BOLD, RED)} "
                f"sur la colonne C{idx_pen_col}."
            )

    if i_choisie is not None and j_choisie is not None:
        print(
            f"On choisit la case "
            f"{color('('+str(i_choisie)+','+str(j_choisie)+')', BOLD, CYAN)} "
            "comme cellule d'allocation (coût minimal sur la ligne/colonne choisie)."
        )


# ============================================================
#  Coût total
# ============================================================
def calculer_cout_transport(allocations, couts, afficher=False):
    """
    Calcule le coût total d'une matrice d'allocations.
    """
    n = len(allocations)
    m = len(allocations[0])
    cout_total = 0

    for i in range(n):
        for j in range(m):
            cout_total += allocations[i][j] * couts[i][j]

    if afficher:
        print("\n===== COÛT TOTAL =====")
        print(f"💰 Coût = {cout_total}")

    return cout_total


# ============================================================
#  Méthode de Balas-Hammer (Vogel)
# ============================================================
def methode_balas_hammer(couts, offre, demande, verbose=True):
    """
    Méthode de Balas-Hammer (Vogel) pour construire une solution initiale.
    Retourne x (allocations) et basis.
    """
    couts = deepcopy(couts)
    offre = offre[:]
    demande = demande[:]

    n = len(offre)
    m = len(demande)

    if sum(offre) != sum(demande):
        raise ValueError("Problème non équilibré : somme(offre) != somme(demande)")

    x = [[0 for _ in range(m)] for _ in range(n)]
    lignes_actives = [True] * n
    colonnes_actives = [True] * m

    # Afficher la matrice X initiale vide
    if verbose:
        afficher_etat_balas(x, offre, demande)

    def calculer_penalites():
        penalites_lignes = [None] * n
        penalites_colonnes = [None] * m

        # Lignes
        for i in range(n):
            if not lignes_actives[i] or offre[i] == 0:
                penalites_lignes[i] = -1
                continue

            couts_ligne_i = [
                couts[i][j]
                for j in range(m)
                if colonnes_actives[j] and demande[j] > 0
            ]
            if len(couts_ligne_i) == 0:
                penalites_lignes[i] = -1
            elif len(couts_ligne_i) == 1:
                penalites_lignes[i] = couts_ligne_i[0]
            else:
                couts_ligne_i.sort()
                penalites_lignes[i] = couts_ligne_i[1] - couts_ligne_i[0]

        # Colonnes
        for j in range(m):
            if not colonnes_actives[j] or demande[j] == 0:
                penalites_colonnes[j] = -1
                continue

            couts_col_j = [
                couts[i][j]
                for i in range(n)
                if lignes_actives[i] and offre[i] > 0
            ]
            if len(couts_col_j) == 0:
                penalites_colonnes[j] = -1
            elif len(couts_col_j) == 1:
                penalites_colonnes[j] = couts_col_j[0]
            else:
                couts_col_j.sort()
                penalites_colonnes[j] = couts_col_j[1] - couts_col_j[0]

        return penalites_lignes, penalites_colonnes

    iteration = 1
    while any(o > 0 for o in offre) and any(d > 0 for d in demande):
        penalites_lignes, penalites_colonnes = calculer_penalites()

        max_pl = max(penalites_lignes)
        max_pc = max(penalites_colonnes)
        max_pg = max(max_pl, max_pc)

        if max_pg < 0:
            break

        meilleures_lignes = [i for i, p in enumerate(penalites_lignes) if p == max_pg]
        meilleures_colonnes = [j for j, p in enumerate(penalites_colonnes) if p == max_pg]

        choisir_ligne = len(meilleures_lignes) > 0 and (max_pl >= max_pc)

        # Choix ligne/colonne + case d'allocation
        i_choisie = j_choisie = None
        idx_pen_ligne = idx_pen_col = None

        if choisir_ligne:
            idx_pen_ligne = meilleures_lignes[0]
            i_choisie = idx_pen_ligne

            cout_min = inf
            j_choisie = None
            for j in range(m):
                if colonnes_actives[j] and demande[j] > 0:
                    if couts[i_choisie][j] < cout_min:
                        cout_min = couts[i_choisie][j]
                        j_choisie = j
        else:
            idx_pen_col = meilleures_colonnes[0]
            j_choisie = idx_pen_col

            cout_min = inf
            i_choisie = None
            for i in range(n):
                if lignes_actives[i] and offre[i] > 0:
                    if couts[i][j_choisie] < cout_min:
                        cout_min = couts[i][j_choisie]
                        i_choisie = i

        quantite = min(offre[i_choisie], demande[j_choisie])

        if verbose:
            print(f"\n=== Itération Balas-Hammer #{iteration} ===")
            afficher_etat_balas(
                x, offre, demande,
                penalites_lignes, penalites_colonnes,
                i_choisie=i_choisie, j_choisie=j_choisie,
                idx_pen_ligne=idx_pen_ligne, idx_pen_col=idx_pen_col,
                delta_max=max_pg, choisir_ligne=choisir_ligne
            )
            print(
                f"\n→ Allocation possible dans la case ({i_choisie}, {j_choisie}) : "
                f"min(offre, demande) = {quantite}"
            )

        x[i_choisie][j_choisie] = quantite

        offre[i_choisie] -= quantite
        demande[j_choisie] -= quantite

        if offre[i_choisie] == 0:
            lignes_actives[i_choisie] = False
        if demande[j_choisie] == 0:
            colonnes_actives[j_choisie] = False

        iteration += 1

    basis = [[(x[i][j] > 0) for j in range(m)] for i in range(n)]
    return x, basis


# ============================================================
#  Chargement des données depuis fichier texte
# ============================================================
def charger_fichier(path):
    with open(path, "r") as f:
        lignes = [l.strip() for l in f.readlines()]

    n, m = map(int, lignes[0].split())
    couts = []
    provisions = []

    line_index = 1
    for _ in range(n):
        vals = list(map(int, lignes[line_index].split()))
        couts.append(vals[:m])
        provisions.append(vals[m])
        line_index += 1

    commandes = list(map(int, lignes[line_index].split()))
    return couts, provisions, commandes

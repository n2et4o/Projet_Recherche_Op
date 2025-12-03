from collections import deque
from math import inf
from copy import deepcopy
import random
import time
import contextlib
import os
import sys
import matplotlib.pyplot as plt
import csv
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
#  Affichages génériques (utile si tu veux débugger)
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
    header = "      " + " ".join(f"{'C' + str(j):>6}" for j in range(m))
    print(header)
    print("      " + "-" * (6 * m))

    for i in range(n):
        cellules = []
        for j in range(m):
            val = formatter(mat[i][j])
            cellules.append(f"{val:>6}")
        print(f"{'S' + str(i):>3} |" + "".join(cellules))


def afficher_matrice_transport(titre, mat, provisions, commandes, highlight=None):
    """
    Affiche une matrice de transport (coûts ou quantités)
    """
    n = len(mat)
    m = len(mat[0]) if n > 0 else 0

    print(f"\n=== {titre} ===")

    header = "      |" + "".join(f"{('C' + str(j)):>6}" for j in range(m)) + " |  Prov"
    print(header)
    print("      " + "-" * (6 * m + 11))

    for i in range(n):
        row_cells = []
        for j in range(m):
            val = mat[i][j]
            txt = f"{val:>6}"
            if highlight is not None and (i, j) == highlight:
                txt = color(txt, BOLD, CYAN)
            row_cells.append(txt)
        prov_txt = f"{provisions[i]:>5}"
        print(f"{'S' + str(i):>3} |" + "".join(row_cells) + " |" + prov_txt)

    print("      " + "-" * (6 * m + 11))

    cmd_cells = "".join(f"{d:>6}" for d in commandes)
    print(" Cmd  |" + cmd_cells)


def afficher_couts(cost, provisions=None, commandes=None):
    if provisions is not None and commandes is not None:
        afficher_matrice_transport("Matrice des coûts (Cij)", cost, provisions, commandes)
    else:
        afficher_matrice_labels("Matrice des coûts (Cij)", cost, formatter=lambda v: v)


def afficher_quantites(x, provisions=None, commandes=None):
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
    n = len(delta)
    m = len(delta[0])

    if entering is None:
        print("\nTous les couts marginaux Delta_ij sont >= 0 sur les cases non basiques :")
        print("La base actuelle est OPTIMALE.")
    else:
        i_e, j_e = entering
        val_e = delta[i_e][j_e]
        print(
            "\n"
            f"Arete entrante choisie : ({i_e}, {j_e}) avec Delta_ij = "
            f"{color(val_e, BOLD, RED)} (le plus negatif parmi les cases non basiques)."
        )

    print("\n=== Couts marginaux Delta = c - c* (Delta<0 en rouge, arete entrante en rouge gras) ===")
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
    visited = set()

    for start in graph:
        if start not in visited:
            queue = deque([(start, None)])

            while queue:
                node, parent = queue.popleft()

                if node in visited:
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
    n = len(x)
    m = len(x[0])

    equations = []
    for i in range(n):
        for j in range(m):
            if x[i][j] > 0 or basis[i][j] is True:
                equations.append((f"S{i}", f"C{j}", cost[i][j]))

    if not equations:
        raise ValueError("Aucune équation basique trouvée pour calculer les potentiels.")

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
                print(f"  ->  E({C}) = E({S}) - {cij} = {E[C]}")
                changed = True
            elif E[C] is not None and E[S] is None:
                E[S] = E[C] + cij
                print(f"  ->  E({S}) = E({C}) + {cij} = {E[S]}")
                changed = True

    print("\nPotentiels finaux :")
    for s in sorted(E.keys()):
        mark = " (référence)" if s == root else ""
        print(f"  E({s}) = {E[s]}{mark}")
    print()

    return E


def compute_potential_costs(x, E):
    n = len(x)
    m = len(x[0])
    c_star = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            c_star[i][j] = E[f"S{i}"] - E[f"C{j}"]

    return c_star


def compute_reduced_costs(cost, c_star):
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
    n = len(x)
    m = len(x[0])

    best = None
    best_val = 0  # on cherche le plus négatif

    for i in range(n):
        for j in range(m):
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
    i0, j0 = entering
    n = len(x)
    m = len(x[0])

    temp_basis = [row[:] for row in basis]
    temp_basis[i0][j0] = True

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

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

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
    Applique le deplacement de quantite sur le cycle.
    cycle = [ ((i,j), '+'), ((i,j), '-'), ... ]
    """
    theta = float("inf")

    for (i, j), sign in cycle:
        if sign == '-':
            theta = min(theta, x[i][j])

    print(f"\n> Maximisation sur le cycle avec theta = {theta}\n")
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
            f"  x[{i},{j}] {color(op_symb, s_col)}= {theta} : {old} -> {x[i][j]}"
        )

    return x


# ============================================================
#  Réparation d'une base dégénérée
# ============================================================
def repair_degenerate_base(x, basis, cost, graph, visited):
    """
    Repare une base degeneree non connexe en ajoutant une case basique
    de cout minimal reliant deux composantes.
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
        print("Sommets non relies a la base :", problematic)
        if isolated:
            print("Parmi eux, sommets totalement isoles :", isolated)

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
            "\nOn cherche une case (i,j) de cout minimal reliant ces composantes."
        )
        print(
            f"On ajoute la case basique ({best_i}, {best_j}) "
            f"de cout {cost[best_i][best_j]} pour reparer la base."
        )
        basis[best_i][best_j] = True
        added = (best_i, best_j)

    return basis, added


# ============================================================
#  Méthode du marche-pied complète
# ============================================================
def marche_pied(x, basis, cost):
    """
    Methode du marche-pied complete.
    Ameliore la base jusqu'a atteindre l'optimalite.
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

        # 2. Dégénérescence : on répare jusqu'à ce que le graphe soit connexe
        while True:
            print("\n> Test de connexité...")
            connexe, visited = is_connected(graph)
            if connexe:
                print(color("Graphe connexe : toutes les lignes et colonnes sont reliées.", GREEN))
                break
            else:
                print(color("Graphe NON connexe : certaines lignes/colonnes sont isolées.", YELLOW))
                print(color("[WARNING] Base dégénérée : tentative de réparation...", YELLOW))
                basis, added = repair_degenerate_base(x, basis, cost, graph, visited)
                if added is None:
                    print(color("Impossible de réparer davantage la base.", RED))
                    break
                print(
                    "La base est réparée en ajoutant la case "
                    f"{color(added, BOLD, CYAN)}."
                )
                afficher_basis(basis)
                graph = build_graph(x, basis)

        # 3. (optionnel) info sur les cycles
        print("> Test d'acyclicité...")
        acyclique = is_acyclic(graph)
        if acyclique:
            print(color("Graphe sans cycle parasite (acyclique).", GREEN))
        else:
            print(color("Cycle détecté dans la base (dégénérescence possible).", YELLOW))

        # 3. Potentiels
        print("\n> Calcul des potentiels :")
        E = compute_potentials(x, cost, basis)

        # 4. Couts potentiels c*
        c_star = compute_potential_costs(x, E)
        afficher_c_star(c_star)

        # 5. Couts marginaux Delta + arete entrante
        delta = compute_reduced_costs(cost, c_star)
        entering = find_entering_arc(x, basis, delta)
        afficher_delta(delta, basis, entering)

        if entering is None:
            print("\n[OK] Solution optimale atteinte !")
            afficher_quantites(x)
            return x, basis

        print(f"\n> Arete entrante retenue : {color(entering, BOLD, RED)}")

        # 7. Cycle correspondant
        cycle = build_cycle_for_entering_arc(x, basis, entering)

        print("\nCycle (positions du cycle avec signes) :")
        for (i, j), sign in cycle:
            col = GREEN if sign == '+' else RED
            print(f"   ({i}, {j}) ({color(sign, col)})")

        # 8. Maximisation
        x = maximize_on_cycle(x, cycle)
        print("\nVoici les matrices mises a jour apres deplacement sur le cycle :")
        afficher_quantites(x)

        # 9. Mise a jour de la base
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
    n = len(x)
    m = len(x[0]) if n > 0 else 0

    print("\n--- Etat courant de la methode Balas-Hammer ---")

    header = "      |" + "".join(f"{('C'+str(j)):>6}" for j in range(m)) + " |  Prov | Delta_ligne"
    print(header)
    print("      " + "-" * (6 * m + 19))

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

    print("      " + "-" * (6 * m + 19))

    cmd_cells = "".join(f"{d:>6}" for d in demande)
    print(" Cmd  |" + cmd_cells + " |" + " " * 7)

    if penalites_colonnes is not None:
        line = "Delta_col |"
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

    if delta_max is not None:
        if choisir_ligne and idx_pen_ligne is not None:
            print(
                f"\nLa penalite maximale est Delta = {color(delta_max, BOLD, RED)} "
                f"sur la ligne S{idx_pen_ligne}."
            )
        elif not choisir_ligne and idx_pen_col is not None:
            print(
                f"\nLa penalite maximale est Delta = {color(delta_max, BOLD, RED)} "
                f"sur la colonne C{idx_pen_col}."
            )

    if i_choisie is not None and j_choisie is not None:
        print(
            f"On choisit la case "
            f"{color('('+str(i_choisie)+','+str(j_choisie)+')', BOLD, CYAN)} "
            "comme cellule d'allocation (cout minimal sur la ligne/colonne choisie)."
        )


# ============================================================
#  Coût total
# ============================================================
def calculer_cout_transport(allocations, couts, afficher=False):
    n = len(allocations)
    m = len(allocations[0])
    cout_total = 0

    for i in range(n):
        for j in range(m):
            cout_total += allocations[i][j] * couts[i][j]

    if afficher:
        print("\n===== COÛT TOTAL =====")
        print(f"Cost = {cout_total}")

    return cout_total


# ============================================================
#  Méthode de Balas-Hammer (Vogel)
# ============================================================
def methode_balas_hammer(couts, offre, demande, verbose=True):
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

    if verbose:
        afficher_etat_balas(x, offre, demande)

    def calculer_penalites():
        penalites_lignes = [None] * n
        penalites_colonnes = [None] * m

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
#  MÉTHODE NORD-OUEST (implémentation simple)
# ============================================================
def methode_nord_ouest(couts, offre, demande, verbose=False):
    """
    Construction d'une solution initiale par la méthode du coin Nord-Ouest.
    Retourne (x, basis).
    """
    couts = deepcopy(couts)
    offre = offre[:]
    demande = demande[:]

    n = len(offre)
    m = len(demande)

    if sum(offre) != sum(demande):
        raise ValueError("Problème non équilibré : somme(offre) != somme(demande)")

    x = [[0 for _ in range(m)] for _ in range(n)]
    basis = [[False for _ in range(m)] for _ in range(n)]

    i = 0
    j = 0

    iteration = 1
    while i < n and j < m:
        q = min(offre[i], demande[j])
        x[i][j] = q
        basis[i][j] = True

        if verbose:
            print(f"[Nord-Ouest] Itération {iteration}: on alloue {q} en ({i},{j})")
        iteration += 1

        offre[i] -= q
        demande[j] -= q

        if offre[i] == 0 and i < n - 1:
            i += 1
        elif demande[j] == 0 and j < m - 1:
            j += 1
        else:
            # cas dégénérés (fin de tableau) : on sort
            i += 1
            j += 1

    return x, basis


# ============================================================
#  Génération de problèmes de transport aléatoires
# ============================================================
def generer_probleme_transport(n, low=1, high=100):
    """
    Génère un problème de transport de taille n x n :
      - couts : matrice A (a_ij) avec entiers entre [low, high]
      - provisions P_i et commandes C_j équilibrées via une matrice temp.
    """
    couts = [
        [random.randint(low, high) for _ in range(n)]
        for _ in range(n)
    ]

    temp = [
        [random.randint(low, high) for _ in range(n)]
        for _ in range(n)
    ]

    provisions = [sum(temp[i][j] for j in range(n)) for i in range(n)]
    commandes = [sum(temp[i][j] for i in range(n)) for j in range(n)]

    return couts, provisions, commandes


# ============================================================
#  Contexte pour couper les prints pendant les mesures
# ============================================================
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        try:
            sys.stdout = devnull
            yield
        finally:
            sys.stdout = old_stdout


# ============================================================
#  Mesure des temps sur 100 instances par n
# ============================================================
def lancer_experiences(
        n_values=(10, 40, 100, 400, 1000, 4000, 10000),
        nb_repetitions=100
):
    """
    Pour chaque n, génère nb_repetitions problèmes aléatoires
    et mesure :
      - θ_NO (Nord-Ouest)
      - θ_BH (Balas-Hammer)
      - t_NO (marche-pied avec Nord-Ouest)
      - t_BH (marche-pied avec Balas-Hammer)
    """
    resultats = {
        "theta_NO": {n: [] for n in n_values},
        "theta_BH": {n: [] for n in n_values},
        "t_NO": {n: [] for n in n_values},
        "t_BH": {n: [] for n in n_values},
    }

    for n in n_values:
        print(f"\n========== n = {n} ==========")
        for k in range(nb_repetitions):
            print(f"  → répétition {k + 1}/{nb_repetitions}")

            # génération d'un nouveau problème aléatoire
            couts, provisions, commandes = generer_probleme_transport(n)

            # 1) Nord-Ouest
            print("     [1] Nord-Ouest...", end="", flush=True)
            t0 = time.perf_counter()
            x_NO, basis_NO = methode_nord_ouest(
                deepcopy(couts),
                provisions[:],
                commandes[:],
                verbose=False
            )
            t1 = time.perf_counter()
            resultats["theta_NO"][n].append(t1 - t0)
            print(f" OK ({t1 - t0:.6f} s)")

            # 2) Balas-Hammer
            print("     [2] Balas-Hammer...", end="", flush=True)
            t0 = time.perf_counter()
            with suppress_stdout():
                x_BH, basis_BH = methode_balas_hammer(
                    deepcopy(couts),
                    provisions[:],
                    commandes[:],
                    verbose=False
                )
            t1 = time.perf_counter()
            resultats["theta_BH"][n].append(t1 - t0)
            print(f" OK ({t1 - t0:.6f} s)")

            # 3) Marche-pied à partir de Nord-Ouest
            print("     [3] Marche-pied (NO)...", end="", flush=True)
            t0 = time.perf_counter()
            with suppress_stdout():
                marche_pied(
                    deepcopy(x_NO),
                    deepcopy(basis_NO),
                    couts
                )
            t1 = time.perf_counter()
            resultats["t_NO"][n].append(t1 - t0)
            print(f" OK ({t1 - t0:.6f} s)")

            # 4) Marche-pied à partir de Balas-Hammer
            print("     [4] Marche-pied (BH)...", end="", flush=True)
            t0 = time.perf_counter()
            with suppress_stdout():
                marche_pied(
                    deepcopy(x_BH),
                    deepcopy(basis_BH),
                    couts
                )
            t1 = time.perf_counter()
            resultats["t_BH"][n].append(t1 - t0)
            print(f" OK ({t1 - t0:.6f} s)")

    return resultats


# ============================================================
#  Tracés et analyses
# ============================================================
def tracer_nuage_points(resultats, cle, titre):
    plt.figure()
    for n, valeurs in resultats[cle].items():
        xs = [n] * len(valeurs)
        ys = valeurs
        plt.scatter(xs, ys, s=10)
    plt.xlabel("n")
    plt.ylabel("Temps (s)")
    plt.title(titre)
    plt.grid(True)
    plt.show()


def calculer_max_par_n(resultats, cle):
    return {n: max(valeurs) for n, valeurs in resultats[cle].items()}


def tracer_courbe_max(max_par_n, titre):
    n_list = sorted(max_par_n.keys())
    y_list = [max_par_n[n] for n in n_list]
    plt.figure()
    plt.plot(n_list, y_list, marker="o")
    plt.xlabel("n")
    plt.ylabel("Temps max (s)")
    plt.title(titre)
    plt.grid(True)
    plt.show()


def ajouter_combinaisons(resultats):
    theta_plus_t_NO = {}
    theta_plus_t_BH = {}

    for n in resultats["theta_NO"].keys():
        list_NO = []
        list_BH = []
        for k in range(len(resultats["theta_NO"][n])):
            total_NO = resultats["theta_NO"][n][k] + resultats["t_NO"][n][k]
            total_BH = resultats["theta_BH"][n][k] + resultats["t_BH"][n][k]
            list_NO.append(total_NO)
            list_BH.append(total_BH)
        theta_plus_t_NO[n] = list_NO
        theta_plus_t_BH[n] = list_BH

    resultats["theta_plus_t_NO"] = theta_plus_t_NO
    resultats["theta_plus_t_BH"] = theta_plus_t_BH


def calculer_rapport_NO_sur_BH(resultats):
    rapports = {}
    rapports_max = {}

    for n in resultats["theta_plus_t_NO"].keys():
        vals_NO = resultats["theta_plus_t_NO"][n]
        vals_BH = resultats["theta_plus_t_BH"][n]

        Rn = []
        for vNO, vBH in zip(vals_NO, vals_BH):
            if vBH == 0:
                R = float("inf")
            else:
                R = vNO / vBH
            Rn.append(R)

        rapports[n] = Rn
        rapports_max[n] = max(Rn)

    return {
        "rapports": rapports,
        "rapports_max": rapports_max
    }


def tracer_rapports_max(rapports_max):
    n_list = sorted(rapports_max.keys())
    y_list = [rapports_max[n] for n in n_list]
    plt.figure()
    plt.plot(n_list, y_list, marker="o")
    plt.xlabel("n")
    plt.ylabel("max ( (θ_NO + t_NO) / (θ_BH + t_BH) )")
    plt.title("Comparaison NO vs BH dans le pire des cas")
    plt.grid(True)
    plt.show()


# ============================================================
#  Sauvegarde des résultats dans des fichiers CSV
# ============================================================

def sauvegarder_resultats_bruts(resultats, nom_fichier="resultats_bruts.csv"):
    """
    Sauvegarde, pour chaque n et chaque repetition k, les temps :
      - theta_NO, theta_BH, t_NO, t_BH
      - theta_NO + t_NO
      - theta_BH + t_BH
      - rapport (theta_NO + t_NO) / (theta_BH + t_BH)
    dans un fichier CSV.
    """
    # On s'assure que les combinaisons sont présentes
    if "theta_plus_t_NO" not in resultats or "theta_plus_t_BH" not in resultats:
        ajouter_combinaisons(resultats)

    with open(nom_fichier, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "n",
            "rep",
            "theta_NO",
            "theta_BH",
            "t_NO",
            "t_BH",
            "theta_plus_t_NO",
            "theta_plus_t_BH",
            "rapport_NO_sur_BH"
        ])

        for n in sorted(resultats["theta_NO"].keys()):
            vals_theta_NO = resultats["theta_NO"][n]
            vals_theta_BH = resultats["theta_BH"][n]
            vals_t_NO = resultats["t_NO"][n]
            vals_t_BH = resultats["t_BH"][n]
            vals_tot_NO = resultats["theta_plus_t_NO"][n]
            vals_tot_BH = resultats["theta_plus_t_BH"][n]

            nb_rep = len(vals_theta_NO)

            for k in range(nb_rep):
                thNO = vals_theta_NO[k]
                thBH = vals_theta_BH[k]
                tNO = vals_t_NO[k]
                tBH = vals_t_BH[k]
                totNO = vals_tot_NO[k]
                totBH = vals_tot_BH[k]
                if totBH == 0:
                    rapport = float("inf")
                else:
                    rapport = totNO / totBH

                writer.writerow([
                    n,
                    k,
                    thNO,
                    thBH,
                    tNO,
                    tBH,
                    totNO,
                    totBH,
                    rapport
                ])


def sauvegarder_maximums(resultats, nom_fichier="resultats_max.csv"):
    """
    Sauvegarde, pour chaque n, les temps maximaux (pire des cas) :
      - max_theta_NO, max_theta_BH, max_t_NO, max_t_BH
      - max(theta_NO + t_NO), max(theta_BH + t_BH)
      - max rapport (theta_NO + t_NO) / (theta_BH + t_BH)
    """
    # S'assurer que les combinaisons et rapports sont calculés
    if "theta_plus_t_NO" not in resultats or "theta_plus_t_BH" not in resultats:
        ajouter_combinaisons(resultats)
    comp = calculer_rapport_NO_sur_BH(resultats)

    max_theta_NO = calculer_max_par_n(resultats, "theta_NO")
    max_theta_BH = calculer_max_par_n(resultats, "theta_BH")
    max_t_NO = calculer_max_par_n(resultats, "t_NO")
    max_t_BH = calculer_max_par_n(resultats, "t_BH")
    max_theta_plus_t_NO = calculer_max_par_n(resultats, "theta_plus_t_NO")
    max_theta_plus_t_BH = calculer_max_par_n(resultats, "theta_plus_t_BH")
    max_rapport = comp["rapports_max"]

    with open(nom_fichier, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "n",
            "max_theta_NO",
            "max_theta_BH",
            "max_t_NO",
            "max_t_BH",
            "max_theta_plus_t_NO",
            "max_theta_plus_t_BH",
            "max_rapport_NO_sur_BH"
        ])

        for n in sorted(resultats["theta_NO"].keys()):
            writer.writerow([
                n,
                max_theta_NO[n],
                max_theta_BH[n],
                max_t_NO[n],
                max_t_BH[n],
                max_theta_plus_t_NO[n],
                max_theta_plus_t_BH[n],
                max_rapport[n]
            ])
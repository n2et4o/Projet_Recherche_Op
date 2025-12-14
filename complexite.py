from collections import deque
from math import inf
from copy import deepcopy
import random
import time
import matplotlib.pyplot as plt
import csv

# ============================================================================
#  Utils console (couleurs optionnelles pour un affichage plus lisible)
# ============================================================================

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def color(text, *styles):
    return "".join(styles) + str(text) + RESET


# ============================================================================
#  Génération de problèmes de transport aléatoires
# ============================================================================

def generer_probleme_transport(n, low=1, high=100):
    """
    Génère un problème de transport de taille n x n :
      - couts : matrice (a_ij) entiers dans [low, high]
      - provisions P_i = somme_j(temp_ij)
      - commandes C_j = somme_i(temp_ij)
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


# ============================================================================
#  Méthode Nord-Ouest (solution initiale)
# ============================================================================

def methode_nord_ouest(couts, offre, demande):
    """
    Méthode du coin Nord-Ouest : construit une solution de base faisable.
    Retourne :
      - x : matrice d'allocations
      - basis : matrice booléenne indiquant les variables basiques
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
    while i < n and j < m:
        q = min(offre[i], demande[j])
        x[i][j] = q
        basis[i][j] = True

        offre[i] -= q
        demande[j] -= q

        if offre[i] == 0 and i < n - 1:
            i += 1
        elif demande[j] == 0 and j < m - 1:
            j += 1
        else:
            # fin de tableau (cas dégénéré)
            i += 1
            j += 1

    return x, basis


# ============================================================================
#  Méthode de Balas-Hammer (Vogel) – version silencieuse (verbose=False)
# ============================================================================

def methode_balas_hammer(couts, offre, demande, verbose=False):
    """
    Méthode de Balas-Hammer (Vogel) pour construire une solution initiale.
    Retourne :
      - x : matrice d'allocations
      - basis : matrice booléenne des basiques
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

    def calculer_penalites():
        penalites_lignes = [None] * n
        penalites_colonnes = [None] * m

        # pénalités lignes
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

        # pénalités colonnes
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

        if choisir_ligne:
            i_choisie = meilleures_lignes[0]
            cout_min = inf
            j_choisie = None
            for j in range(m):
                if colonnes_actives[j] and demande[j] > 0:
                    if couts[i_choisie][j] < cout_min:
                        cout_min = couts[i_choisie][j]
                        j_choisie = j
        else:
            j_choisie = meilleures_colonnes[0]
            cout_min = inf
            i_choisie = None
            for i in range(n):
                if lignes_actives[i] and offre[i] > 0:
                    if couts[i][j_choisie] < cout_min:
                        cout_min = couts[i][j_choisie]
                        i_choisie = i

        quantite = min(offre[i_choisie], demande[j_choisie])
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


# ============================================================================
#  Marche-pied – VERSION RAPIDE (optimisée, sans prints)
# ============================================================================

def _build_graph_from_basis(basis):
    """
    Construit le graphe biparti (S_i, C_j) à partir de la matrice basis.
    """
    n = len(basis)
    m = len(basis[0])

    graph = {f"S{i}": [] for i in range(n)}
    graph.update({f"C{j}": [] for j in range(m)})

    for i in range(n):
        for j in range(m):
            if basis[i][j]:
                s = f"S{i}"
                c = f"C{j}"
                graph[s].append(c)
                graph[c].append(s)

    return graph


def _is_connected_fast(graph):
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


def _repair_degenerate_base_fast(x, basis, cost, graph, visited):
    """
    Répare une base non connexe en ajoutant une case basique de coût minimal
    reliant deux composantes. Version sans affichage.
    """
    n = len(x)
    m = len(x[0])

    all_nodes = set(graph.keys())
    problematic = all_nodes - visited

    problematic_S = {int(s[1:]) for s in problematic if s.startswith("S")}
    problematic_C = {int(c[1:]) for c in problematic if c.startswith("C")}

    visited_S = {int(s[1:]) for s in visited if s.startswith("S")}
    visited_C = {int(c[1:]) for c in visited if c.startswith("C")}

    best_i, best_j = None, None
    best_cost = float("inf")

    for i in range(n):
        for j in range(m):
            if basis[i][j]:
                continue
            if x[i][j] != 0:
                continue

            connects = (
                (i in problematic_S and j in visited_C) or
                (i in visited_S and j in problematic_C)
            )
            if not connects:
                continue

            if cost[i][j] < best_cost:
                best_cost = cost[i][j]
                best_i, best_j = i, j

    if best_i is not None:
        basis[best_i][best_j] = True
        return basis, (best_i, best_j)

    return basis, None


def _compute_potentials_fast(cost, basis):
    """
    Calcule les potentiels u[i] et v[j] de manière ROBUSTE
    (gère toutes les composantes connexes).
    """
    n = len(cost)
    m = len(cost[0])

    u = [None] * n
    v = [None] * m

    # Graphe biparti
    adj = {f"S{i}": [] for i in range(n)}
    adj.update({f"C{j}": [] for j in range(m)})

    for i in range(n):
        for j in range(m):
            if basis[i][j]:
                adj[f"S{i}"].append((f"C{j}", cost[i][j]))
                adj[f"C{j}"].append((f"S{i}", cost[i][j]))

    # On parcourt TOUTES les composantes
    for i_start in range(n):
        if u[i_start] is not None:
            continue

        # référence arbitraire
        u[i_start] = 0
        queue = deque([f"S{i_start}"])

        while queue:
            node = queue.popleft()

            if node.startswith("S"):
                i = int(node[1:])
                for neigh, cij in adj[node]:
                    j = int(neigh[1:])
                    if v[j] is None:
                        v[j] = u[i] - cij
                        queue.append(neigh)

            else:  # C_j
                j = int(node[1:])
                for neigh, cij in adj[node]:
                    i = int(neigh[1:])
                    if u[i] is None:
                        u[i] = v[j] + cij
                        queue.append(neigh)

    return u, v



def _compute_reduced_costs_fast(cost, u, v):
    n = len(cost)
    m = len(cost[0])
    delta = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            c_star = u[i] - v[j]
            delta[i][j] = cost[i][j] - c_star
    return delta


def _find_entering_arc_fast(basis, delta):
    """
    Trouve l'arête entrante (i,j) correspondant au Delta_ij le plus négatif
    parmi les cases non basiques. Retourne None si tous Delta_ij >= 0.
    """
    n = len(delta)
    m = len(delta[0])

    best = None
    best_val = 0.0
    for i in range(n):
        for j in range(m):
            if basis[i][j]:
                continue
            if delta[i][j] < best_val:
                best_val = delta[i][j]
                best = (i, j)
    return best


def _build_cycle_for_entering_arc_fast(basis, entering):
    """
    Construit le cycle associé à l'arête entrante.
    Retourne une liste [((i,j), signe), ...] où signe ∈ {'+','-'}.
    """
    i0, j0 = entering
    n = len(basis)
    m = len(basis[0])

    # base temporaire avec l'arête entrante ajoutée
    temp_basis = [row[:] for row in basis]
    temp_basis[i0][j0] = True

    # construire le graphe
    graph = {f"S{i}": [] for i in range(n)}
    graph.update({f"C{j}": [] for j in range(m)})

    for i in range(n):
        for j in range(m):
            if temp_basis[i][j]:
                s = f"S{i}"
                c = f"C{j}"
                graph[s].append(c)
                graph[c].append(s)

    start = f"S{i0}"
    target = f"C{j0}"

    # BFS pour trouver un chemin de start à target
    queue = deque([start])
    parent = {start: None}

    while queue:
        node = queue.popleft()
        for neigh in graph[node]:
            # éviter d'utiliser immédiatement l'arête directe (S_i0, C_j0)
            if node == f"S{i0}" and neigh == f"C{j0}":
                continue
            if neigh not in parent:
                parent[neigh] = node
                queue.append(neigh)
                if neigh == target:
                    queue.clear()
                    break

    if target not in parent:
        raise ValueError("Impossible de construire le cycle pour l'arête entrante.")

    # reconstituer le chemin
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()  # start -> target

    # convertir en positions (i,j)
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

    # insérer l'arête entrante en première position
    cycle_positions.insert(0, (i0, j0))

    cycle = []
    sign = '+'
    for pos in cycle_positions:
        cycle.append((pos, sign))
        sign = '-' if sign == '+' else '+'

    return cycle


def _maximize_on_cycle_fast(x, basis, cycle):
    """
    Applique le déplacement de quantité sur le cycle.
    Met à jour x et basis (choix d'une variable sortante parmi les '-').
    """
    # chercher theta (min x[i][j] sur les cases '-' avec x[i][j] > 0)
    theta = float("inf")
    leaving_pos = None

    for (i, j), sign in cycle:
        if sign == '-' and x[i][j] > 0:
            if x[i][j] < theta:
                theta = x[i][j]
                leaving_pos = (i, j)

    if theta == float("inf") or theta == 0:
        # pas de mouvement possible (dégénérescence bloquante)
        return False

    # appliquer le mouvement
    for (i, j), sign in cycle:
        if sign == '+':
            x[i][j] += theta
        else:
            x[i][j] -= theta

    # mise à jour de la base : l'arête entrante devient basique
    entering_pos = cycle[0][0]
    ei, ej = entering_pos
    basis[ei][ej] = True

    # l'arête sortante devient non basique
    if leaving_pos is not None and leaving_pos != entering_pos:
        li, lj = leaving_pos
        basis[li][lj] = False

    return True


def marche_pied_rapide(x, basis, cost, max_iterations_factor=10):
    """
    Méthode du marche-pied (stepping-stone) optimisée :
      - pas d'affichage,
      - base réparée si non connexe,
      - potentiels calculés via BFS,
      - cycle construit pour chaque arête entrante,
      - mise à jour de x et de la base.
    """
    n = len(x)
    m = len(x[0])
    max_iterations = max_iterations_factor * n * m
    iteration = 0

    while True:
        iteration += 1
        if iteration > max_iterations:
            # sécurité : on arrête si trop d'itérations
            break

        # 1. construire le graphe de base
        graph = _build_graph_from_basis(basis)

        # 2. vérifier la connexité, sinon réparer la base
        while True:
            connexe, visited = _is_connected_fast(graph)
            if connexe:
                break
            basis, added = _repair_degenerate_base_fast(x, basis, cost, graph, visited)
            if added is None:
                break
            graph = _build_graph_from_basis(basis)

        # 3. calcul des potentiels
        u, v = _compute_potentials_fast(cost, basis)

        # 4. coûts réduits
        delta = _compute_reduced_costs_fast(cost, u, v)

        # 5. recherche de l'arête entrante
        entering = _find_entering_arc_fast(basis, delta)
        if entering is None:
            # optimalité atteinte
            break

        # 6. construire le cycle et maximiser
        cycle = _build_cycle_for_entering_arc_fast(basis, entering)
        ok = _maximize_on_cycle_fast(x, basis, cycle)
        if not ok:
            # pas de progression possible => on arrête pour éviter boucle infinie
            break

    return x, basis


# ============================================================================
#  Une expérience (pour un n donné)
# ============================================================================

def _une_experience(n):
    """
    Effectue UNE répétition complète pour une taille n :
      - génération d'un problème de transport aléatoire,
      - calcul de θ_NO (Nord-Ouest),
      - calcul de θ_BH (Balas-Hammer),
      - calcul de t_NO (marche-pied à partir de NO),
      - calcul de t_BH (marche-pied à partir de BH).
    """
    # génération du problème
    couts, provisions, commandes = generer_probleme_transport(n)

    # 1) Nord-Ouest
    t0 = time.perf_counter()
    x_NO, basis_NO = methode_nord_ouest(
        couts,
        provisions,
        commandes
    )
    t1 = time.perf_counter()
    theta_NO = t1 - t0

    # 2) Balas-Hammer
    t0 = time.perf_counter()
    x_BH, basis_BH = methode_balas_hammer(
        couts,
        provisions,
        commandes,
        verbose=False
    )
    t1 = time.perf_counter()
    theta_BH = t1 - t0

    # 3) Marche-pied à partir de Nord-Ouest
    t0 = time.perf_counter()
    marche_pied_rapide(
        deepcopy(x_NO),
        deepcopy(basis_NO),
        couts
    )
    t1 = time.perf_counter()
    t_NO = t1 - t0

    # 4) Marche-pied à partir de Balas-Hammer
    t0 = time.perf_counter()
    marche_pied_rapide(
        deepcopy(x_BH),
        deepcopy(basis_BH),
        couts
    )
    t1 = time.perf_counter()
    t_BH = t1 - t0

    return theta_NO, theta_BH, t_NO, t_BH


# ============================================================================
#  Lancement des expériences – 100 répétitions par n, MONO-COEUR
# ============================================================================

def lancer_experiences(
        n_values=(10, 40, 100, 400, 1000, 4000, 10000),
        nb_repetitions=100
):
    """
    Pour chaque n dans n_values, effectue nb_repetitions expériences SEQUENTIELLES
    (mono-coeur, conforme au sujet) et stocke les temps :
      - θ_NO, θ_BH, t_NO, t_BH
    """
    resultats = {
        "theta_NO": {n: [] for n in n_values},
        "theta_BH": {n: [] for n in n_values},
        "t_NO": {n: [] for n in n_values},
        "t_BH": {n: [] for n in n_values},
    }

    for n in n_values:
        print(color(f"\n========== n = {n} ==========", BOLD, CYAN))

        for rep in range(nb_repetitions):
            theta_NO, theta_BH, t_NO, t_BH = _une_experience(n)

            resultats["theta_NO"][n].append(theta_NO)
            resultats["theta_BH"][n].append(theta_BH)
            resultats["t_NO"][n].append(t_NO)
            resultats["t_BH"][n].append(t_BH)

            print(
                f"  Répétition {rep + 1:3d}/{nb_repetitions} : "
                f"θNO={theta_NO:.6f}s, θBH={theta_BH:.6f}s, "
                f"tNO={t_NO:.6f}s, tBH={t_BH:.6f}s"
            )

    return resultats


# ============================================================================
#  Tracés & analyses
# ============================================================================

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
    if "theta_plus_t_NO" not in resultats or "theta_plus_t_BH" not in resultats:
        ajouter_combinaisons(resultats)

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


# ============================================================================
#  Sauvegarde des résultats dans des fichiers CSV
# ============================================================================

def sauvegarder_resultats_bruts(resultats, nom_fichier="resultats_bruts.csv"):
    """
    Sauvegarde, pour chaque n et chaque répétition k, les temps :
      - θ_NO, θ_BH, t_NO, t_BH
      - θ_NO + t_NO
      - θ_BH + t_BH
      - rapport (θ_NO + t_NO) / (θ_BH + t_BH)
    """
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


# ============================================================================
#  Point d'entrée (exemple d'utilisation)
# ============================================================================

if __name__ == "__main__":
    # ⚠ Attention : les grandes tailles (1000, 4000, 10000) avec 100 répétitions
    # peuvent être TRES longues et très lourdes en mémoire.
    # Pour tester rapidement, commence par :
    # n_values = (10, 40, 100)
    n_values = (10, 40, 100)  # adapte ensuite vers (10,40,100,400,1000,4000,10000)
    nb_repetitions = 100

    resultats = lancer_experiences(n_values=n_values, nb_repetitions=nb_repetitions)

    # Ajout des combinaisons θ + t
    ajouter_combinaisons(resultats)

    # Nuages de points
    tracer_nuage_points(resultats, "theta_NO", "Nuage de points θ_NO(n)")
    tracer_nuage_points(resultats, "theta_BH", "Nuage de points θ_BH(n)")
    tracer_nuage_points(resultats, "t_NO", "Nuage de points t_NO(n)")
    tracer_nuage_points(resultats, "t_BH", "Nuage de points t_BH(n)")
    tracer_nuage_points(resultats, "theta_plus_t_NO", "Nuage de points (θ_NO + t_NO)(n)")
    tracer_nuage_points(resultats, "theta_plus_t_BH", "Nuage de points (θ_BH + t_BH)(n)")

    # Enveloppes max (pire des cas)
    max_theta_NO = calculer_max_par_n(resultats, "theta_NO")
    max_theta_BH = calculer_max_par_n(resultats, "theta_BH")
    max_t_NO = calculer_max_par_n(resultats, "t_NO")
    max_t_BH = calculer_max_par_n(resultats, "t_BH")
    max_theta_plus_t_NO = calculer_max_par_n(resultats, "theta_plus_t_NO")
    max_theta_plus_t_BH = calculer_max_par_n(resultats, "theta_plus_t_BH")

    tracer_courbe_max(max_theta_NO, "Pire des cas θ_NO(n)")
    tracer_courbe_max(max_theta_BH, "Pire des cas θ_BH(n)")
    tracer_courbe_max(max_t_NO, "Pire des cas t_NO(n)")
    tracer_courbe_max(max_t_BH, "Pire des cas t_BH(n)")
    tracer_courbe_max(max_theta_plus_t_NO, "Pire des cas (θ_NO + t_NO)(n)")
    tracer_courbe_max(max_theta_plus_t_BH, "Pire des cas (θ_BH + t_BH)(n)")

    # Comparaison NO vs BH (pire des cas)
    comp = calculer_rapport_NO_sur_BH(resultats)
    tracer_rapports_max(comp["rapports_max"])

    # Sauvegarde CSV
    sauvegarder_resultats_bruts(resultats, "resultats_bruts.csv")
    sauvegarder_maximums(resultats, "resultats_max.csv")

    print(color("\nFin des expériences et des tracés.", GREEN, BOLD))


from math import inf
from copy import deepcopy

def methode_balas_hammer(couts, offre, demande, verbose=True):
    """
    Méthode de Balas-Hammer (Vogel) pour construire une solution initiale.

    Paramètres
    ----------
    couts   : matrice des coûts unitaires (n x m)
    offre   : liste des provisions (longueur n)
    demande : liste des commandes (longueur m)

    Retour
    ------
    x      : matrice des quantités allouées (n x m)
    basis  : matrice booléenne (n x m), True si case basique (x_ij > 0)
    """

    # Copies pour ne pas modifier les originaux
    couts   = deepcopy(couts)
    offre   = offre[:]      # provisions
    demande = demande[:]    # commandes

    n = len(offre)
    m = len(demande)

    if sum(offre) != sum(demande):
        raise ValueError("Problème non équilibré : somme(offre) != somme(demande)")

    # Matrice des allocations (solution initiale)
    x = [[0 for _ in range(m)] for _ in range(n)]

    lignes_actives   = [True] * n
    colonnes_actives = [True] * m

    def calculer_penalites():
        """Calcule les pénalités de chaque ligne et colonne."""
        penalites_lignes   = [None] * n
        penalites_colonnes = [None] * m

        # --- lignes ---
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

        # --- colonnes ---
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

        if verbose:
            print(f"\n=== Itération Balas-Hammer #{iteration} ===")
            print("Offres restantes   :", offre)
            print("Demandes restantes :", demande)
            print("Pénalités lignes   :", penalites_lignes)
            print("Pénalités colonnes :", penalites_colonnes)

        max_pl = max(penalites_lignes)
        max_pc = max(penalites_colonnes)
        max_pg = max(max_pl, max_pc)

        if max_pg < 0:  # sécurité
            break

        meilleures_lignes = [i for i, p in enumerate(penalites_lignes) if p == max_pg]
        meilleures_colonnes = [j for j, p in enumerate(penalites_colonnes) if p == max_pg]

        choisir_ligne = len(meilleures_lignes) > 0 and (max_pl >= max_pc)

        if choisir_ligne:
            i_choisie = meilleures_lignes[0]
            if verbose:
                print(f"Ligne(s) pénalité max : {meilleures_lignes}")

            cout_min = inf
            j_choisie = None
            for j in range(m):
                if colonnes_actives[j] and demande[j] > 0:
                    if couts[i_choisie][j] < cout_min:
                        cout_min = couts[i_choisie][j]
                        j_choisie = j
        else:
            j_choisie = meilleures_colonnes[0]
            if verbose:
                print(f"Colonne(s) pénalité max : {meilleures_colonnes}")

            cout_min = inf
            i_choisie = None
            for i in range(n):
                if lignes_actives[i] and offre[i] > 0:
                    if couts[i][j_choisie] < cout_min:
                        cout_min = couts[i][j_choisie]
                        i_choisie = i

        quantite = min(offre[i_choisie], demande[j_choisie])
        x[i_choisie][j_choisie] = quantite

        if verbose:
            print(
                f"→ Case ({i_choisie}, {j_choisie}) "
                f"coût {couts[i_choisie][j_choisie]} : allocation {quantite}"
            )

        offre[i_choisie]   -= quantite
        demande[j_choisie] -= quantite

        if offre[i_choisie] == 0:
            lignes_actives[i_choisie] = False
        if demande[j_choisie] == 0:
            colonnes_actives[j_choisie] = False

        iteration += 1

    # --- Construction de basis : basique là où x_ij > 0 ---
    basis = [[(x[i][j] > 0) for j in range(m)] for i in range(n)]

    return x, basis

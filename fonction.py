
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


def afficher_matrice(couts, valeurs, provisions, commandes):
    n = len(couts)
    m = len(couts[0])

    # -------- 1) Calcul LARGEUR DE COLONNE UNIQUE --------
    contenus = []

    # en-têtes colonnes
    contenus += ["C" + str(j+1) for j in range(m)]
    contenus.append("Provisions")
    contenus.append("Commande")

    # étiquettes de lignes
    contenus += [f"P{i+1}" for i in range(n)]

    # coûts
    for row in couts:
        contenus += [str(c) for c in row]

    # valeurs (non vides)
    for row in valeurs:
        contenus += [str(v) for v in row if str(v) != "" and v is not None]

    # provisions et commandes
    contenus += [str(p) for p in provisions]
    contenus += [str(c) for c in commandes]
    contenus.append(str(sum(commandes)))

    max_len = max(len(x) for x in contenus)
    col = max_len + 2

    # -------- 2) Fonctions utilitaires --------
    def ligne_sep():
        return "+" + "+".join("-" * col for _ in range(m + 2)) + "+"

    def print_row(cells, left_align_cols=None):
        # left_align_cols = liste des colonnes à aligner à gauche
        row = []
        for idx, cell in enumerate(cells):
            if left_align_cols and idx in left_align_cols:
                row.append(str(cell).ljust(col))     # ALIGNÉ A GAUCHE
            else:
                row.append(str(cell).center(col))   # CENTRÉ
        print("|" + "|".join(row) + "|")

    # -------- 3) En-tête --------
    print(ligne_sep())
    header = [""] + ["C" + str(j+1) for j in range(m)] + ["Provisions"]
    print_row(header)
    print(ligne_sep())

    # -------- 4) Lignes P_i --------
    for i in range(n):

        # Colonnes Pᵢ × Cⱼ = alignées à gauche
        # Indices 1 à m uniquement
        left_cols = list(range(1, m + 1))

        # Ligne des coûts
        ligne_cost = [f"P{i+1}"] + [str(x) for x in couts[i]] + [str(provisions[i])]
        print_row(ligne_cost, left_align_cols=left_cols)

        # Ligne des valeurs → centrées
        ligne_val = [""] + [
            "" if str(valeurs[i][j]) in ("", "None") else str(valeurs[i][j])
            for j in range(m)
        ] + [""]
        print_row(ligne_val)

        print(ligne_sep())

    # -------- 5) Ligne Commande --------
    ligne_cmd = ["Commande"] + [str(c) for c in commandes] + [str(sum(commandes))]
    print_row(ligne_cmd)
    print(ligne_sep())


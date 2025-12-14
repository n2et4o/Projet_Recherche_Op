from FONCTIONS_TEST import*
from Nord_Ouest import *

def main():

    print("=======================================")
    print("   PROJET TRANSPORT — MARCHE-PIED")
    print("=======================================")

    # ============================================================
    # 1. Sélection du fichier
    # ============================================================
    #path = input("Entrez le chemin du fichier .txt : ")
    path = "./Matrice/M1.txt"
    try:
        couts, provisions, commandes = charger_fichier(path)
    except Exception as e:
        print("\n Erreur lors du chargement du fichier :", e)
        return

    # ============================================================
    # 2. Affichage des données
    # ============================================================
    print("\n=== Données chargées ===")
    #
    afficher_couts(couts, provisions, commandes)
    #valeurs = [["" for _ in row] for row in couts]
    valeurs,basis = nord_ouest(provisions, commandes)
    afficher_matrice(couts, valeurs, provisions, commandes)

    print("Provisions :", provisions)
    print("Commandes  :", commandes)

    # ============================================================
    # 3. Méthode de Balas-Hammer
    # ============================================================
    print("\n=======================================")
    print("   ETAPE 1 — MÉTHODE BALAS-HAMMER")
    print("=======================================")

    x_init, basis_init = methode_balas_hammer(
        couts,
        provisions,
        commandes,
        verbose=True
    )

    print("\n=== Allocation initiale (Balas-Hammer) ===")
    afficher_quantites(x_init, provisions, commandes)
    print("We are basis ==========================================================")
    afficher_basis(basis_init)

    # ============================================================
    # 4. Méthode du marche-pied
    # ============================================================
    print("\n=======================================")
    print("   ETAPE 2 — MÉTHODE DU MARCHE-PIED")
    print("=======================================")

    x_opt, basis_opt = marche_pied(
        x_init,
        basis_init,
        couts
    )
    print("Nord-Ouest MP  +=============================================================")
    val_opt, basiss_opt = marche_pied(
        valeurs,
        basis,
        couts
    )

    # ============================================================
    # 5. Coût total optimal
    # ============================================================
    print("\n=======================================")
    print("   SOLUTION OPTIMALE — COÛT TOTAL")
    print("=======================================")

    cout_opt = calculer_cout_transport(val_opt, couts, afficher=True)

    print("\n FIN DU PROGRAMME — SOLUTION OPTIMALE ATTEINTE ")

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


if __name__ == "__main__":
    main()
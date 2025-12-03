from FONCTIONS_TEST import*

def main():

    print("=======================================")
    print("   PROJET TRANSPORT — MARCHE-PIED")
    print("=======================================")

    # ============================================================
    # 1. Sélection du fichier
    # ============================================================
    path = input("\n🔎 Entrez le chemin du fichier .txt : ")

    try:
        couts, provisions, commandes = charger_fichier(path)
    except Exception as e:
        print("\n❌ Erreur lors du chargement du fichier :", e)
        return

    # ============================================================
    # 2. Affichage des données
    # ============================================================
    print("\n=== Données chargées ===")
    afficher_couts(couts, provisions, commandes)

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

    # ============================================================
    # 5. Coût total optimal
    # ============================================================
    print("\n=======================================")
    print("   SOLUTION OPTIMALE — COÛT TOTAL")
    print("=======================================")

    cout_opt = calculer_cout_transport(x_opt, couts, afficher=True)

    print("\n🔥 FIN DU PROGRAMME — SOLUTION OPTIMALE ATTEINTE 🔥")


if __name__ == "__main__":
    main()
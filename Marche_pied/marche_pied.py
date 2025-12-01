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

        # ---------------------------------------------------
        # AFFICHAGE : Proposition actuelle
        # ---------------------------------------------------
        afficher_tableau("Matrice X (proposition)", x)
        afficher_tableau("Base (basis)", basis)

        # ---------------------------------------------------
        # 1. Graphe de la base
        # ---------------------------------------------------
        print("\n> Graphe de la base :")
        graph = build_graph(x, basis)

        # ---------------------------------------------------
        # 2. Dégénérescence
        # ---------------------------------------------------
        print("\n> Test de connexité...")
        connexe, visited = is_connected(graph)

        print("> Test d'acyclicité...")
        acyclique = is_acyclic(graph)

        if not connexe or not acyclique:
            print("⚠️  Base dégénérée : on répare...")
            basis = repair_degenerate_base(x, basis, cost, graph, visited)

            afficher_tableau("Base réparée", basis)

            graph = build_graph(x, basis)

        # ---------------------------------------------------
        # 3. Potentiels
        # ---------------------------------------------------
        print("\n> Calcul des potentiels :")
        E = compute_potentials(x, cost, basis)
        print("Potentiels :", E)

        # ---------------------------------------------------
        # 4. Coûts potentiels c*
        # ---------------------------------------------------
        c_star = compute_potential_costs(x, E)
        afficher_tableau("Coûts potentiels (c*)", c_star)

        # ---------------------------------------------------
        # 5. Coûts marginaux Δ
        # ---------------------------------------------------
        delta = compute_reduced_costs(cost, c_star)
        afficher_tableau("Coûts marginaux Δ = c - c*", delta)

        # ---------------------------------------------------
        # 6. Arête entrante
        # ---------------------------------------------------
        entering = find_entering_arc(x, delta)

        if entering is None:
            print("\n✅ Solution optimale atteinte !")
            afficher_tableau("Solution finale", x)
            return x, basis

        print(f"Arête entrante : {entering}")

        # ---------------------------------------------------
        # 7. Cycle correspondant
        # ---------------------------------------------------
        cycle = build_cycle_for_entering_arc(x, basis, entering)

        print("Cycle :")
        for pos, sign in cycle:
            print(f"   {pos} ({sign})")

        # ---------------------------------------------------
        # 8. Maximisation
        # ---------------------------------------------------
        x = maximize_on_cycle(x, cycle)
        afficher_tableau("Matrice X après maximisation", x)

        # ---------------------------------------------------
        # 9. Mise à jour de la base
        # ---------------------------------------------------
        ei, ej = entering
        basis[ei][ej] = True

        afficher_tableau("Base mise à jour", basis)

        iteration += 1


 

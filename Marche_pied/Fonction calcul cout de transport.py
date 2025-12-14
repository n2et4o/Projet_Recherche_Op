def calculer_cout_transport(allocations, couts, afficher=False):
    """
    Calcule le coût total d'une matrice d'allocations.
    Option : afficher=True pour afficher une table formatée.
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

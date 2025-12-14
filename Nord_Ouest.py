def nord_ouest(provision, commande):
    """
    Implémente la méthode du Nord-Ouest pour le problème de transport
    sans modifier les listes d'entrée.

    Retourne :
        - valeurs : matrice des quantités allouées
        - basis   : matrice bool indiquant les cases de base
    """

    # Copies locales pour éviter de modifier les originaux
    provision_local = provision.copy()
    commande_local = commande.copy()

    n = len(provision_local)
    m = len(commande_local)

    # Matrice d'allocations
    valeurs = [[0 for _ in range(m)] for _ in range(n)]

    i, j = 0, 0

    while i < n and j < m:
        q = min(provision_local[i], commande_local[j])
        valeurs[i][j] = q

        provision_local[i] -= q
        commande_local[j] -= q

        if provision_local[i] == 0:
            i += 1
        if commande_local[j] == 0:
            j += 1

    # Construction de la basis comme Balas-Hammer
    basis = [[(valeurs[i][j] > 0) for j in range(m)] for i in range(n)]

    return valeurs, basis


# # Exemple d'utilisation
# provisions = [100, 100]       # Pi
# commandes = [100, 100]        # Cj

# solution = nord_ouest(provisions[:], commandes[:])  # [:] pour éviter de modifier les listes originales
# print("Solution Nord-Ouest :")
# for ligne in solution:
#     print(ligne)
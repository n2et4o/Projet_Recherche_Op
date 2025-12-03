def nord_ouest(provisions, commandes):
    """
    Implémente la méthode du Nord-Ouest pour le problème de transport.
    
    :param provisions: liste des quantités disponibles chez chaque fournisseur (Pi)
    :param commandes: liste des quantités demandées par chaque client (Cj)
    :return: matrice de transport (allocation des quantités)
    """
    n = len(provisions)   # nombre de fournisseurs
    m = len(commandes)    # nombre de clients
    
    # Initialisation de la matrice de transport avec des zéros
    allocation = [[0 for _ in range(m)] for _ in range(n)]
    
    i, j = 0, 0  # indices de départ (coin nord-ouest)
    
    # Tant qu'il reste des provisions et des commandes
    while i < n and j < m:
        # On affecte la quantité minimale possible
        q = min(provisions[i], commandes[j])
        allocation[i][j] = q
        
        # Mise à jour des provisions et commandes restantes
        provisions[i] -= q
        commandes[j] -= q
        
        # Si le fournisseur i est épuisé, on passe au suivant
        if provisions[i] == 0:
            i += 1
        # Si le client j est satisfait, on passe au suivant
        if commandes[j] == 0:
            j += 1
    
    return allocation


# # Exemple d'utilisation
# provisions = [100, 100]       # Pi
# commandes = [100, 100]        # Cj

# solution = nord_ouest(provisions[:], commandes[:])  # [:] pour éviter de modifier les listes originales
# print("Solution Nord-Ouest :")
# for ligne in solution:
#     print(ligne)
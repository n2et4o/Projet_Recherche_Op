from math import inf          # On importe 'inf' (infini) pour initialiser des coûts très grands
from copy import deepcopy     # deepcopy permet de copier des structures complexes sans les modifier en place


def methode_balas_hammer(couts, offre, demande, verbose=True):
    """
    Méthode de Balas-Hammer (méthode des pénalités / de Vogel)
    pour construire une solution initiale du problème de transport.

    Paramètres
    ----------
    couts  : liste de listes (n x m)
        Matrice des coûts unitaires a_{i,j}.
    offre  : liste de longueur n
        Provisions P_i pour chaque fournisseur.
    demande : liste de longueur m
        Commandes C_j pour chaque client.

    Retour
    ------
    allocation : liste de listes (n x m)
        Matrice b_{i,j} des quantités transportées (solution initiale).
    """

    # On fait une copie profonde de la matrice de coûts pour ne pas modifier l'originale
    couts = deepcopy(couts)

    # On copie également les listes offre et demande (copie superficielle suffisante pour des listes simples)
    offre = offre[:]      
    demande = demande[:]  

    # n = nombre de lignes (fournisseurs)
    n = len(offre)

    # m = nombre de colonnes (clients)
    m = len(demande)

    # On vérifie que le problème de transport est équilibré : somme des offres = somme des demandes
    if sum(offre) != sum(demande):
        # Si ce n'est pas le cas, on lève une erreur
        raise ValueError("Problème non équilibré : somme(offre) != somme(demande)")

    # On initialise la matrice d'allocation (b_{i,j}) avec des 0 (aucune quantité allouée au départ)
    allocation = [[0 for _ in range(m)] for _ in range(n)]

    # On crée une liste pour indiquer quelles lignes sont encore actives (non saturées)
    lignes_actives = [True] * n

    # On crée une liste pour indiquer quelles colonnes sont encore actives (non saturées)
    colonnes_actives = [True] * m

    # On définit une fonction interne pour calculer les pénalités de chaque ligne et colonne
    def calculer_penalites():
        """
        Calcule les pénalités pour chaque ligne et chaque colonne.
        Retourne deux listes : penalites_lignes, penalites_colonnes.
        """

        # Liste des pénalités de chaque ligne (initialement None)
        penalites_lignes = [None] * n

        # Liste des pénalités de chaque colonne (initialement None)
        penalites_colonnes = [None] * m

        # --- Calcul des pénalités pour les lignes ---
        for i in range(n):  # On parcourt chaque ligne i
            # Si la ligne n'est plus active ou l'offre restante est 0, on ignore cette ligne
            if not lignes_actives[i] or offre[i] == 0:
                penalites_lignes[i] = -1   # -1 signifie "à ignorer"
                continue                   # On passe à la ligne suivante

            # On récupère les coûts de la ligne i uniquement pour les colonnes actives ayant une demande > 0
            couts_ligne_i = [
                couts[i][j]
                for j in range(m)
                if colonnes_actives[j] and demande[j] > 0
            ]

            # Si aucun coût disponible (toutes les colonnes sont saturées pour cette ligne)
            if len(couts_ligne_i) == 0:
                penalites_lignes[i] = -1   # On ignore aussi
            # S'il n'y a qu'une seule case possible dans cette ligne
            elif len(couts_ligne_i) == 1:
                # La pénalité est juste ce coût (pas de différence possible)
                penalites_lignes[i] = couts_ligne_i[0]
            else:
                # On trie la liste des coûts pour trouver les deux plus petits
                couts_ligne_i.sort()
                # Pénalité = deuxième plus petit coût - plus petit coût
                penalites_lignes[i] = couts_ligne_i[1] - couts_ligne_i[0]

        # --- Calcul des pénalités pour les colonnes ---
        for j in range(m):  # On parcourt chaque colonne j
            # Si la colonne n'est plus active ou la demande restante est 0, on ignore cette colonne
            if not colonnes_actives[j] or demande[j] == 0:
                penalites_colonnes[j] = -1  # -1 signifie "à ignorer"
                continue                    # On passe à la colonne suivante

            # On récupère les coûts de la colonne j uniquement pour les lignes actives ayant une offre > 0
            couts_colonne_j = [
                couts[i][j]
                for i in range(n)
                if lignes_actives[i] and offre[i] > 0
            ]

            # Si aucun coût disponible (toutes les lignes sont saturées pour cette colonne)
            if len(couts_colonne_j) == 0:
                penalites_colonnes[j] = -1  # On ignore aussi
            # S'il n'y a qu'une seule case possible dans cette colonne
            elif len(couts_colonne_j) == 1:
                # La pénalité est juste ce coût
                penalites_colonnes[j] = couts_colonne_j[0]
            else:
                # On trie la liste des coûts pour trouver les deux plus petits
                couts_colonne_j.sort()
                # Pénalité = deuxième plus petit coût - plus petit coût
                penalites_colonnes[j] = couts_colonne_j[1] - couts_colonne_j[0]

        # On renvoie les deux listes de pénalités
        return penalites_lignes, penalites_colonnes

    # Compteur d'itération (juste pour l'affichage si verbose=True)
    iteration = 1

    # Boucle principale : on continue tant qu'il reste de l'offre et de la demande
    while any(o > 0 for o in offre) and any(d > 0 for d in demande):

        # À chaque itération, on calcule les pénalités
        penalites_lignes, penalites_colonnes = calculer_penalites()

        # Si l'affichage détaillé est activé, on montre l'état courant
        if verbose:
            print(f"\n=== Itération Balas-Hammer #{iteration} ===")
            print("Offres restantes   :", offre)
            print("Demandes restantes :", demande)
            print("Pénalités lignes   :", penalites_lignes)
            print("Pénalités colonnes :", penalites_colonnes)

        # On récupère la pénalité maximale parmi toutes les lignes
        max_penalite_ligne = max(penalites_lignes)

        # On récupère la pénalité maximale parmi toutes les colonnes
        max_penalite_colonne = max(penalites_colonnes)

        # La plus grande pénalité globale (entre lignes et colonnes)
        max_penalite_globale = max(max_penalite_ligne, max_penalite_colonne)

        # Si toutes les pénalités sont < 0, cela signifie qu'on ne peut plus rien allouer (cas anormal)
        if max_penalite_globale < 0:
            # On sort de la boucle par sécurité
            break

        # On récupère la liste des indices de lignes qui atteignent la pénalité maximale
        meilleures_lignes = [
            i for i, p in enumerate(penalites_lignes) if p == max_penalite_globale
        ]

        # On récupère la liste des indices de colonnes qui atteignent la pénalité maximale
        meilleures_colonnes = [
            j for j, p in enumerate(penalites_colonnes) if p == max_penalite_globale
        ]

        # Règle de choix :
        #  - si au moins une ligne a la pénalité max
        #  - ET que max_penalite_ligne >= max_penalite_colonne
        #  alors on choisit une ligne, sinon on choisit une colonne
        choisir_ligne = len(meilleures_lignes) > 0 and (max_penalite_ligne >= max_penalite_colonne)

        # Si on a décidé de choisir une ligne
        if choisir_ligne:
            # On prend la première ligne dans la liste des meilleures (on pourrait changer la règle de tie-break)
            i_choisie = meilleures_lignes[0]

            # Affichage (optionnel) des lignes candidates si verbose=True
            if verbose:
                print(f"Ligne(s) de pénalité maximale : {meilleures_lignes}")

            # Dans cette ligne, on cherche la colonne (active) avec le coût le plus faible
            cout_min = inf      # On initialise le coût minimum à l'infini
            j_choisie = None    # On mémorisera ici l'indice de la meilleure colonne

            # On parcourt toutes les colonnes
            for j in range(m):
                # On ne considère que les colonnes actives et ayant une demande > 0
                if colonnes_actives[j] and demande[j] > 0:
                    # Si le coût dans la case (i_choisie, j) est plus petit que le coût minimal courant
                    if couts[i_choisie][j] < cout_min:
                        cout_min = couts[i_choisie][j]  # On met à jour le coût minimal
                        j_choisie = j                   # On mémorise l'indice de cette colonne

        # Sinon, on a décidé de choisir une colonne
        else:
            # On prend la première colonne dans la liste des meilleures
            j_choisie = meilleures_colonnes[0]

            # Affichage (optionnel) des colonnes candidates si verbose=True
            if verbose:
                print(f"Colonne(s) de pénalité maximale : {meilleures_colonnes}")

            # Dans cette colonne, on cherche la ligne (active) avec le coût le plus faible
            cout_min = inf      # On initialise le coût minimum à l'infini
            i_choisie = None    # On mémorisera ici l'indice de la meilleure ligne

            # On parcourt toutes les lignes
            for i in range(n):
                # On ne considère que les lignes actives et ayant une offre > 0
                if lignes_actives[i] and offre[i] > 0:
                    # Si le coût dans la case (i, j_choisie) est plus petit que le coût minimal courant
                    if couts[i][j_choisie] < cout_min:
                        cout_min = couts[i][j_choisie]  # On met à jour le coût minimal
                        i_choisie = i                   # On mémorise l'indice de cette ligne

        # À ce stade, on a choisi une case (i_choisie, j_choisie) dans laquelle allouer une quantité
        # La quantité allouée est le minimum entre l'offre de la ligne et la demande de la colonne
        quantite = min(offre[i_choisie], demande[j_choisie])

        # On inscrit cette quantité dans la matrice d'allocation
        allocation[i_choisie][j_choisie] = quantite

        # Affichage (optionnel) du choix réalisé
        if verbose:
            print(
                f"→ Choix de la case (ligne {i_choisie}, colonne {j_choisie}) "
                f"de coût {couts[i_choisie][j_choisie]} : allocation {quantite}"
            )

        # On met à jour l'offre de la ligne i_choisie
        offre[i_choisie] -= quantite

        # On met à jour la demande de la colonne j_choisie
        demande[j_choisie] -= quantite

        # Si l'offre de la ligne est maintenant 0, on désactive cette ligne
        if offre[i_choisie] == 0:
            lignes_actives[i_choisie] = False

        # Si la demande de la colonne est maintenant 0, on désactive cette colonne
        if demande[j_choisie] == 0:
            colonnes_actives[j_choisie] = False

        # On passe à l'itération suivante
        iteration += 1

    # Quand la boucle se termine, l'allocation contient la solution initiale (b_{i,j})
    return allocation

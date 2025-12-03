
#la matrice allocations est la matrice issue de l'élaboration d'une proposotion initiale en balas_hammer ou North-west,
#elle sera mise à jour après la méthode de marche pied dans la proposition finale
def calculer_cout_transport(allocations, couts):
    cout_total = 0
    for i in range(len(allocations)):
        for j in range(len(allocations[0])):
            cout_total += allocations[i][j] * couts[i][j]
    return cout_total
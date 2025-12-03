
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


def afficher_couts(cost):
    n = len(cost)
    m = len(cost[0])

    print("\n=== MATRICE DES COUTS (Cij) ===")
    print("      " + " ".join([f"C{j}" for j in range(m)]))

    for i in range(n):
        ligne = [f"{cost[i][j]:>4}" for j in range(m)]
        print(f"S{i} | " + " ".join(ligne))


def afficher_quantites(x):
    n = len(x)
    m = len(x[0])

    print("\n=== MATRICE DES QUANTITES (Xij) ===")
    print("      " + " ".join([f"C{j}" for j in range(m)]))

    for i in range(n):
        ligne = [f"{x[i][j]:>4}" for j in range(m)]
        print(f"S{i} | " + " ".join(ligne))



def afficher_basis(basis):
    n = len(basis)
    m = len(basis[0])

    print("\n=== MATRICE BASIS (True = basique) ===")
    print("      " + " ".join([f"C{j}" for j in range(m)]))

    for i in range(n):
        ligne = [ "  B " if basis[i][j] else "  . " for j in range(m) ]
        print(f"S{i} | " + "".join(ligne))




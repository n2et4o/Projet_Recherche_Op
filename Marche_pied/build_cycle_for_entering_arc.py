def build_cycle_for_entering_arc(x, basis, entering):
    """
    Construit le cycle associé à l'arête entrante (i,j)
    et retourne la liste des cases du cycle avec signes + / -.

    Paramètres
    ----------
    x        : matrice des quantités
    basis    : matrice booléenne des cases basiques
    entering : tuple (i,j) provenant de find_entering_arc

    Retour
    ------
    cycle : liste [ ((i,j), '+'), ((i,j), '-'), ... ]
    """

    i0, j0 = entering
    n = len(x)
    m = len(x[0])

    # --- 1) Construire un graphe temporaire avec l'arête entrante ---
    temp_basis = [row[:] for row in basis]
    temp_basis[i0][j0] = True   # on ajoute l’arête entrante

    # graphe biparti basique
    graph = {}
    for i in range(n):
        graph[f"S{i}"] = []
    for j in range(m):
        graph[f"C{j}"] = []

    for i in range(n):
        for j in range(m):
            if temp_basis[i][j] is True or x[i][j] > 0:
                graph[f"S{i}"].append(f"C{j}")
                graph[f"C{j}"].append(f"S{i}")

    # --- 2) Trouver un chemin S_i0 → C_j0 qui n'utilise pas (i0,j0) directement ---
    start = f"S{i0}"
    target = f"C{j0}"

    from collections import deque
    queue = deque([start])
    parent = {start: None}

    while queue:
        node = queue.popleft()

        for neigh in graph[node]:

            # on interdit l'arête directe entrante (pour forcer un cycle)
            if node == f"S{i0}" and neigh == f"C{j0}":
                continue

            if neigh not in parent:
                parent[neigh] = node
                queue.append(neigh)

                if neigh == target:
                    queue.clear()
                    break

    # --- 3) Reconstruire le chemin (cycle) ---
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()  # de S_i0 vers C_j0

    # convertir Sx / Cy en indices (i,j)
    cycle_positions = []
    for k in range(len(path) - 1):
        a = path[k]
        b = path[k + 1]

        if a.startswith("S"):
            i = int(a[1:])
            j = int(b[1:])
        else:
            j = int(a[1:])
            i = int(b[1:])

        cycle_positions.append((i, j))

    # fermer le cycle avec l'arête entrante
    cycle_positions.insert(0, (i0, j0))

    # --- 4) Affecter les signes + / - ---
    cycle = []
    sign = '+'
    for pos in cycle_positions:
        cycle.append((pos, sign))
        sign = '-' if sign == '+' else '+'

    return cycle
 

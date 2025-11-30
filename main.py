import networkx as nx
import matplotlib.pyplot as plt
#from Interface_graphiques import *
from fonction import *

def dessiner_matrice(sources, destinations, couts, valeurs):
    G = nx.Graph()

    # Ajouter les noeuds sources
    for s in sources:
        G.add_node(s, bipartite=0)

    # Ajouter les noeuds destinations
    for c in destinations:
        G.add_node(c, bipartite=1)

    # Ajouter les arcs
    for i, s in enumerate(sources):
        for j, c in enumerate(destinations):
            if valeurs[i][j] != 0:  # seulement les arcs alloués
                etiquette = f"{valeurs[i][j]} ({couts[i][j]})"
                G.add_edge(s, c, label=etiquette)

    # Position des nœuds en deux colonnes
    pos = {}

    # Colonnes alignées verticalement
    for i, s in enumerate(sources):
        pos[s] = (0, len(sources) - i)

    for j, c in enumerate(destinations):
        pos[c] = (4, len(destinations) - j)

    # Dessiner les nœuds
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=2000,
        node_color="lightgray",
        font_size=12
    )

    # Dessiner les étiquettes des arcs
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12)

    plt.axis('off')
    plt.show()


#"""
# --- utilisation ---
couts, provisions, commandes = charger_fichier("./Matrice/M10.txt")
# valeurs à la même taille que couts
valeurs = [["" for _ in row] for row in couts]
"""valeurs = [
    [00, 100],
    [100, 00]
]"""
afficher_matrice(couts,valeurs, provisions, commandes)

sources = ["P1", "P2"]
destinations = ["C1", "C2"]

#dessiner_matrice(sources, destinations, couts, valeurs)
#"""

"""
state = State()  # Crée un objet contenant les états

# Boucle principale
running = True
while running:
    if state.choosing_options:  # Affichage des options
        draw_options_menu(screen, state)
    elif state.showing_help:  # Affichage de l'aide
        draw_help(screen, state)
    elif state.choosing_matrice:  # Affichage du choix du matrice
        draw_input_box(screen, state)

    elif state.chosen_matrice is not None:  # menu du matrice
        draw_menu(screen, state)
    else:
        draw_menu(screen, state)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if state.showing_help:
                if event.key == pygame.K_BACKSPACE:
                    state.showing_help = False

            # Gestion du choix du matrice
            elif state.choosing_matrice:
                if event.key == pygame.K_RETURN:
                    try:
                        state.chosen_matrice = int(state.input_text)
                        if 1 <= state.chosen_matrice <= 12:
                            state.choosing_matrice = False
                            # récupérer chemin correctement (EXE + IDE)
                            fichier_test = (f"./Matrice/M{state.chosen_matrice}.txt")
                            print(f"Affichage du matrice {state.chosen_matrice}")
                            nb = state.chosen_matrice
                            couts, provisions, commandes = charger_fichier(fichier_test)
                            # valeurs à la même taille que couts
                            # valeurs = [["" for _ in row] for row in couts]
                            valeurs = [
                                [00, 100],
                                [100, 00]
                            ]
                            afficher_matrice(couts, valeurs, provisions, commandes)
                            #trace_execution_floyd(nb, n, m, arcs, Matrice, historique, chemins_log)
                            #afficher_matrice(Matrice)
                            #afficher_matrice_pygame(screen, n, m, arcs, Matrice,nb)


                            state.options = [
                                "Afficher le matrice",
                                "Resolution via Nord-Ouest",
                                "Resolution via Balas-Hammer",
                                "Resolution via marche-pied",
                                "Aide",
                                "Retour au menu principal",
                                "Quitter",
                            ]
                            state.selected = 0
                            state.input_text = ""
                        else:
                            state.input_text = ""
                    except ValueError:
                        state.input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    state.input_text = state.input_text[:-1]
                else:
                    state.input_text += event.unicode

            # Gestion des options
            elif state.choosing_options:
                if event.key == pygame.K_DOWN:
                    state.selected_mode = (state.selected_mode + 1) % len(state.mode_options)
                elif event.key == pygame.K_UP:
                    state.selected_mode = (state.selected_mode - 1) % len(state.mode_options)
                elif event.key == pygame.K_RETURN:
                    state.current_mode = state.mode_options[state.selected_mode]
                    state.current_background_color = MODES[state.current_mode]["background"]
                    state.text_color = MODES[state.current_mode]["text"]
                    state.choosing_options = False
                elif event.key == pygame.K_BACKSPACE:
                    state.choosing_options = False

            # Gestion de la floydification du matrice
            # elif state.choosing_floydification:

            # Gestion du menu principal
            else:
                if event.key == pygame.K_DOWN:
                    state.selected = (state.selected + 1) % len(state.options)
                elif event.key == pygame.K_UP:
                    state.selected = (state.selected - 1) % len(state.options)
                elif event.key == pygame.K_RETURN:
                    if state.chosen_matrice is None:  # Menu principal
                        if state.selected == 0:
                            state.choosing_matrice = True
                        elif state.selected == 1:
                            state.choosing_options = True
                        elif state.selected == 2:
                            state.showing_help = True
                        elif state.selected == 3:  # Quitter
                            running = False

                    elif state.chosen_matrice is not None:
                            if state.selected == 0:
                                print(f"Affichage du matrice {state.chosen_matrice}")
                                afficher_matrice(couts, valeurs, provisions, commandes)

                            elif state.selected == 1:
                                print(f"floydification du matrice {state.chosen_matrice}")
                                 # Pour revenir au menu Pygame
                                #trace_execution_floyd(state.chosen_matrice, n, m, arcs, Matrice, historique, chemins_log)
                                state.choosing_floydification = False
                                state.selected = 0 # Réinitialisation pour afficher le matrice floydifier

                            elif state.selected == 2:
                                state.showing_help = True
                            elif state.selected == 3:
                                state.chosen_matrice = None
                                state.showing_graph = False
                                state.options = ["Choisir un matrice", "Options", "Aide", "Quitter"]
                            elif state.selected == 4:  # Quitter depuis le sous-menu
                                running = False

    state.clock.tick(60)  # Limite à 60 FPS


pygame.quit()
sys.exit()

"""

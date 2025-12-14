from main_test import *
from Interface_graphiques import *




#"""
# --- utilisation ---
#couts, provisions, commandes = charger_fichier("./Matrice/M1.txt")
# valeurs à la même taille que couts
#valeurs = [["" for _ in row] for row in couts]
valeurs = [
    [00, 100],
    [100, 00]
]
#afficher_matrice(couts,valeurs, provisions, commandes)

sources = ["P1", "P2"]
destinations = ["C1", "C2"]

#dessiner_matrice(sources, destinations, couts, valeurs)
#"""



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
                            valeurs = [["" for _ in row] for row in couts]
                            afficher_matrice(couts, valeurs, provisions, commandes)
                            afficher_matrice_transport_pygame(
                                screen,
                                couts,
                                valeurs,
                                provisions,
                                commandes,
                                titre=f"Matrice {state.chosen_matrice}"
                            )

                            afficher_couts(couts, provisions, commandes)
                            #trace_execution_floyd(nb, n, m, arcs, Matrice, historique, chemins_log)


                            state.options = [
                                "Afficher le matrice",
                                "Resolution via Nord-Ouest",
                                "Resolution via Balas-Hammer",
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
                                afficher_matrice_transport_pygame(screen, couts,valeurs,provisions,commandes,titre=f"Matrice {state.chosen_matrice}")


                            elif state.selected == 1:
                                print(f"Resolution via Nord-Ouest de la matrice {state.chosen_matrice}")
                                valeurs, basis = nord_ouest(provisions, commandes)
                                afficher_matrice(couts, valeurs, provisions, commandes)
                                afficher_matrice_transport_pygame(screen,couts,valeurs,provisions,commandes,titre=f"Matrice {state.chosen_matrice}")
                                print("Nord-Ouest MP  +=============================================================")

                                dessiner_graphe_basis_pygame(screen,basis,titre="Graphe biparti de la base")

                                val_opt, basiss_opt = marche_pied(
                                    valeurs,
                                    basis,
                                    couts
                                )

                                # Capture EXACTE de la sortie console
                                texte_balas = print_f(marche_pied, val_opt, basiss_opt, couts, )

                                # Affichage Pygame du texte console
                                afficher_console_pygame(screen, texte_balas,
                                                        titre=f"Marche Pied — Matrice {state.chosen_matrice}")

                                dessiner_graphe_basis_pygame(screen,basiss_opt,titre="Graphe biparti de la base")


                                cout_opt = calculer_cout_transport(val_opt, couts, afficher=True)

                                # Capture EXACTE de la sortie console
                                texte_balas = print_f(calculer_cout_transport, val_opt, couts, afficher=True)

                                # Affichage Pygame du texte console
                                afficher_console_pygame(screen, texte_balas,
                                                        titre=f"Coût optimal — Matrice {state.chosen_matrice}")

                                cout_opt = calculer_cout_transport(val_opt, couts, afficher=True)
                                 # Pour revenir au menu Pygame
                                #trace_execution_floyd(state.chosen_matrice, n, m, arcs, Matrice, historique, chemins_log)
                                state.choosing_floydification = False
                                state.selected = 0 # Réinitialisation pour afficher le matrice

                            elif state.selected == 2:
                                print(f"Resolution via Balas-Hammer de la matrice {state.chosen_matrice}")
                                val_init, basis_init = methode_balas_hammer(couts,provisions,commandes,verbose=True)

                                steps = balas_hammer_steps(couts, provisions, commandes)
                                balas_hammer_pygame(screen, couts, steps)
                                afficher_matrice_transport_pygame(screen,couts,val_init,provisions,commandes,titre=f"Matrice {state.chosen_matrice}")


                                dessiner_graphe_basis_pygame(screen,basis_init,titre="Graphe biparti de la base")

                                # Capture EXACTE de la sortie console
                                texte_balas = print_f(methode_balas_hammer,couts,provisions,commandes,True)

                                # Affichage Pygame du texte console
                                afficher_console_pygame(screen,texte_balas,titre=f"Balas-Hammer — Matrice {state.chosen_matrice}")

                                afficher_matrice(couts, valeurs, provisions, commandes)

                                val_opt, basis_opt = marche_pied(val_init,basis_init,couts)

                                # Capture EXACTE de la sortie console
                                texte_balas = print_f(marche_pied,val_init,basis_init,couts,)

                                # Affichage Pygame du texte console
                                afficher_console_pygame(screen,texte_balas,titre=f"Marche Pied — Matrice {state.chosen_matrice}")

                                dessiner_graphe_basis_pygame(screen,basis_opt,titre="Graphe biparti de la base")

                                cout_opt = calculer_cout_transport(val_opt, couts, afficher=True)

                                # Capture EXACTE de la sortie console
                                texte_balas = print_f(calculer_cout_transport,val_opt, couts, afficher=True)

                                # Affichage Pygame du texte console
                                afficher_console_pygame(screen,texte_balas,titre=f"Coût optimal — Matrice {state.chosen_matrice}")

                                 # Pour revenir au menu Pygame
                                #trace_execution_floyd(state.chosen_matrice, n, m, arcs, Matrice, historique, chemins_log)
                                state.choosing_floydification = False
                                state.selected = 0 # Réinitialisation pour afficher le matrice

                            elif state.selected == 3:
                                state.showing_help = True
                            elif state.selected == 4:
                                state.chosen_matrice = None
                                state.showing_graph = False
                                state.options = ["Choisir un matrice", "Options", "Aide", "Quitter"]
                            elif state.selected == 5:  # Quitter depuis le sous-menu
                                running = False

    state.clock.tick(60)  # Limite à 60 FPS


pygame.quit()
sys.exit()



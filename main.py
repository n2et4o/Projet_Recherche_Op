import sys
import pygame

from main_test import *
from Interface_graphiques import *

# ============================================================
# INITIALISATION
# ============================================================
pygame.init()
screen = pygame.display.set_mode((1400, 800))
pygame.display.set_caption("Méthodes de Transport")

state = State()
running = True

# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
while running:

    # ========================================================
    # MODE MARCHE-PIED (ÉCRAN DÉDIÉ)
    # ========================================================
    if state.mode == "MARCHE_PIED":

        marche_pied_pygame(
            screen,
            state.mp_data[0],  # x_init
            state.mp_data[1],  # basis_init
            state.mp_data[2],  # couts
        )

        # 🔁 RETOUR AU MENU MATRICE
        state.mode = "MENU"
        state.selected = 0
        state.mp_data = None
        continue

    # ========================================================
    # AFFICHAGE DES MENUS
    # ========================================================
    if state.choosing_options:
        draw_options_menu(screen, state)

    elif state.showing_help:
        draw_help(screen, state)

    elif state.choosing_matrice:
        draw_input_box(screen, state)

    else:
        draw_menu(screen, state)

    # ========================================================
    # GESTION DES ÉVÉNEMENTS
    # ========================================================
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            # =========================
            # AIDE
            # =========================
            if state.showing_help:
                if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                    state.showing_help = False

            # =========================
            # CHOIX MATRICE
            # =========================
            elif state.choosing_matrice:

                if event.key == pygame.K_RETURN:
                    try:
                        num = int(state.input_text)
                        if 1 <= num <= 12:
                            state.chosen_matrice = num
                            state.choosing_matrice = False

                            fichier = f"./Matrice/M{num}.txt"
                            couts, provisions, commandes = charger_fichier(fichier)

                            valeurs = [["" for _ in row] for row in couts]

                            afficher_matrice_transport_pygame(
                                screen, couts, valeurs,
                                provisions, commandes,
                                titre=f"Matrice {num}"
                            )

                            state.options = [
                                "Afficher la matrice",
                                "Résolution Nord-Ouest",
                                "Résolution Balas-Hammer",
                                "Aide",
                                "Retour menu principal",
                                "Quitter"
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

            # =========================
            # OPTIONS VISUELLES
            # =========================
            elif state.choosing_options:
                if event.key == pygame.K_UP:
                    state.selected_mode = (state.selected_mode - 1) % len(state.mode_options)
                elif event.key == pygame.K_DOWN:
                    state.selected_mode = (state.selected_mode + 1) % len(state.mode_options)
                elif event.key == pygame.K_RETURN:
                    state.current_mode = state.mode_options[state.selected_mode]
                    state.current_background_color = MODES[state.current_mode]["background"]
                    state.text_color = MODES[state.current_mode]["text"]
                    state.choosing_options = False
                elif event.key == pygame.K_ESCAPE:
                    state.choosing_options = False

            # =========================
            # MENU PRINCIPAL / MATRICE
            # =========================
            else:
                if event.key == pygame.K_UP:
                    state.selected = (state.selected - 1) % len(state.options)

                elif event.key == pygame.K_DOWN:
                    state.selected = (state.selected + 1) % len(state.options)

                elif event.key == pygame.K_RETURN:

                    # -------- MENU PRINCIPAL --------
                    if state.chosen_matrice is None:
                        if state.selected == 0:
                            state.choosing_matrice = True
                        elif state.selected == 1:
                            state.choosing_options = True
                        elif state.selected == 2:
                            state.showing_help = True
                        elif state.selected == 3:
                            running = False

                    # -------- MENU MATRICE --------
                    else:
                        if state.selected == 0:
                            afficher_matrice_transport_pygame(
                                screen, couts, valeurs,
                                provisions, commandes,
                                titre=f"Matrice {state.chosen_matrice}"
                            )

                        elif state.selected == 1:
                            valeurs, basis = nord_ouest(provisions, commandes)
                            afficher_matrice_transport_pygame(
                                screen, couts, valeurs,
                                provisions, commandes,
                                titre="Nord-Ouest"
                            )

                        elif state.selected == 2:
                            # ============================
                            # BALAS → MARCHE-PIED
                            # ============================
                            val_init, basis_init = methode_balas_hammer(
                                couts, provisions, commandes, verbose=True
                            )

                            afficher_matrice_transport_pygame(
                                screen, couts, val_init,
                                provisions, commandes,
                                titre="Solution initiale Balas-Hammer"
                            )

                            # 🔴 DONNÉES TRANSMISES AU MODE MARCHE-PIED
                            state.mp_data = (val_init, basis_init, couts)
                            state.mode = "MARCHE_PIED"

                        elif state.selected == 3:
                            state.showing_help = True

                        elif state.selected == 4:
                            state.chosen_matrice = None
                            state.options = ["Choisir une matrice", "Options", "Aide", "Quitter"]
                            state.selected = 0

                        elif state.selected == 5:
                            running = False

    state.clock.tick(60)

# ============================================================
pygame.quit()
sys.exit()

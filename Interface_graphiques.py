import pygame
import time
import random
import io, contextlib
import re, pygame  # Importation du module pour manipuler les expressions régulières
import sys, os #pour intérargir avec syst d'exploitation - manipuler les répertoires et fichiers
from FONCTIONS_TEST import *

# Initialisation de Pygame
pygame.init()

# Paramètres de la fenêtre
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Projet Recherche Opérationnelle - Résolution des problèmes de transport")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # dossier temporaire PyInstaller
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

backgrounds = [
    pygame.image.load(resource_path("images/background3.png"))
]
backgrounds = [pygame.transform.scale(bg, (WIDTH, HEIGHT)) for bg in backgrounds]

# Variables de gestion du fond
background_index = 0
change_time = pygame.time.get_ticks()

# Couleurs
WHITE = (255, 255, 255)
text_color = (0, 0, 0)
BLUE = (100, 100, 255)

# Police
font = pygame.font.Font(None, 50)
font_help = pygame.font.Font(None, 20)
input_font = pygame.font.Font(None, 40)

# Modes de couleurs
MODES = {
    "Clair": {"background": (255, 255, 255), "text": (0, 0, 0)},
    "Sombre": {"background": (30, 30, 30), "text": (255, 255, 255)},
    "Bleu Nuit": {"background": (10, 10, 50), "text": (200, 200, 255)}
}


class State:
    def __init__(self):

        # Options du menu
        self.options = ["Choisir un matrice", "Options", "Aide", "Quitter"]
        self.selected = 0
        self.choosing_matrice = False
        self.chosen_matrice = None
        self.choosing_options = False
        self.input_text = ""
        self.showing_help = False
        self.clock = pygame.time.Clock()
        self.mode_options = list(MODES.keys())
        self.selected_mode = 0

        # Mode actuel
        self.default_mode = "Bleu Nuit"
        self.current_mode = self.default_mode
        self.current_background_color = MODES[self.current_mode]["background"]
        self.text_color = MODES[self.current_mode]["text"]
        self.running = True

        self.mode = "MENU"  # MENU | BALAS
        self.mp_data = None
        self.page = 0  # écran courant dans Balas
        self.bh_step = 0  # étape Balas-Hammer
        self.pages = []
        self.sub_selected = None


def draw_balas_hammer_step(screen, couts, steps, state):
    step = steps[state.bh_step]

    balas_hammer_pygame(
        screen,
        couts,
        steps
    )
def draw_matrice(screen, couts, val, prov, cmd, titre):
    afficher_matrice_transport_pygame(
        screen,
        couts,
        val,
        prov,
        cmd,
        titre=titre
    )
def draw_graphe(screen, basis, titre):
    dessiner_graphe_basis_pygame(screen, basis, titre=titre)
def draw_console(screen, texte, titre):
    afficher_console_pygame(screen, texte, titre=titre)


def draw_menu(screen,state):
    #Affiche le menu avec le fond dynamique.
    global background_index, change_time

    # Vérifier si 5 secondes sont écoulées
    if pygame.time.get_ticks() - change_time >= 3000:
        background_index = (background_index + 1) % len(backgrounds)
        change_time = pygame.time.get_ticks()  # Mettre à jour le temps

    # Afficher l'image de fond
    screen.blit(backgrounds[background_index], (0, 0))

    # Afficher le titre
    title = font.render("Bienvenue!", True, state.text_color)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))

    # Afficher les options du menu
    for i, option in enumerate(state.options):
        color = BLUE if i == state.selected else state.text_color
        text = font.render(option, True, color)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 200 + i * 60))

    pygame.display.flip()

def draw_help(screen, state):
    screen.fill(state.current_background_color)

    help_text = (
        "Ce programme permet de résoudre un problème de transport en lisant un tableau de contraintes depuis un fichier .txt,"
        " en construisant les matrices de coûts,de provisions et de commandes,"
        " puis en appliquant différents algorithmes de Recherche Opérationnelle pour optimiser le coût total du transport."
        " Il génère et affiche proprement la matrice des coûts, la proposition de transport, "
        "les tables des potentiels et des coûts marginaux, et détecte les cycles ou problèmes de connexité dans le matrice associé afin de les corriger."
        " Le code implémente notamment les méthodes du Nord-Ouest, de Balas-Hammer et du marche-pied avec potentiels,"
        " permettant d’obtenir progressivement une solution optimale tout en offrant une analyse complète des étapes intermédiaires nécessaires à la résolution."
    )

    # Utilisation de render_text_multiline pour gérer l'affichage proprement
    text = render_text_multiline(help_text, font_help, state.text_color, WIDTH - 10)

    y_offset = HEIGHT // 10  # Position de départ
    for line in text:
        rendered_text = font.render(line, True, state.text_color)
        screen.blit(rendered_text, (20, y_offset))  # Légère marge pour éviter d'être collé au bord
        y_offset += 35  # Espacement entre les lignes

    # Ajout du texte de retour en bas de l'écran
    back_text = font.render("Appuyez sur Supprimer (la croix) pour revenir", True, (0, 0, 255))
    screen.blit(back_text, (WIDTH // 2 - back_text.get_width() // 2, HEIGHT - 50))

    pygame.display.flip()

def render_text_multiline(text, font, color, max_width):
    #Découpe un texte trop long en plusieurs lignes.

    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        test_surface = font.render(test_line, True, color)

        if test_surface.get_width() > max_width:
            lines.append(current_line)  # Ajoute la ligne précédente
            current_line = word + " "  # Commence une nouvelle ligne
        else:
            current_line = test_line  # Continue la ligne actuelle

    lines.append(current_line)  # Ajoute la dernière ligne
    return lines

def draw_input_box(screen, state):
    # Liste dynamique des fichiers .txt dans le dossier matrices_Projet
    dossier = resource_path("Matrice")
    fichiers = [f for f in os.listdir(dossier) if f.endswith(".txt")]
    fichiers.sort()  # Optionnel : trie les fichiers
    total = len(fichiers)

    # Affichage de l'instruction dynamique
    screen.fill(state.current_background_color)
    if total > 0:
        text = f"Entrez un numéro entre 1 et {total}:"
    elif total > 12:
        text = f"Le(s) fichier(s) que vous avez creer commence à partir de 0.\n Entrez un numéro entre 1 et {total}:"
    else:
        text = "Aucun fichier .txt trouvé dans matrices_Projet"

    A_text = font.render(text, True, state.text_color)
    screen.blit(A_text, (WIDTH // 2 - A_text.get_width() // 2, HEIGHT // 3))

    input_surface = input_font.render(state.input_text, True, BLUE)
    screen.blit(input_surface, (WIDTH // 2 - input_surface.get_width() // 2, HEIGHT // 2))

    pygame.display.flip()

def draw_matrice_menu(screen,state):
    screen.fill(state.current_background_color)
    title = font.render(f"matrice {state.chosen_matrice}", True, state.text_color)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))
    sub_options = ["Afficher le matrice", "Resolution via Nord-Ouest",
                    "Resolution via Balas-Hammer", "Aide",
                   "Retour au menu principal", "Quitter"]

    for i, option in enumerate(sub_options):
        color = BLUE if i == state.selected else state.text_color
        text = font.render(option, True, color)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 200 + i * 60))

    pygame.display.flip()

def print_f(func, *args, **kwargs):
    #Capture tout ce que la fonction affiche via print() et le retourne.
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        func(*args, **kwargs)
    return output.getvalue().strip()

def draw_options_menu(screen,state):
    screen.fill(state.current_background_color)
    title = font.render("Choisir un Mode:", True, state.text_color)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 50))

    for i, mode in enumerate(state.mode_options):
        color = (100, 100, 255) if i == state.selected_mode else state.text_color
        text = font.render(mode, True, color)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, 200 + i * 60))

    pygame.display.flip()


def afficher_matrice_transport_pygame(screen, couts, valeurs, provisions, commandes, titre="Matrice",
                                      highlight=None):

    n = len(couts)
    m = len(couts[0])

    W, H = screen.get_width(), screen.get_height()

    # =========================
    # Marges écran
    # =========================
    margin_left = 40
    margin_top = 120
    margin_right = 40
    margin_bottom = 60

    # =========================
    # Polices FIXES (important)
    # =========================
    font_cost = pygame.font.SysFont("consolas", 18)    # coût discret
    font_value = pygame.font.SysFont("consolas", 18)   # valeur visible
    font_label = pygame.font.SysFont("consolas", 18)
    title_font = pygame.font.SysFont("consolas", 26, bold=True)
    help_font = pygame.font.SysFont("consolas", 20)

    # =========================
    # Outils texte
    # =========================
    def text_width(text):
        return font_label.render(text, True, (0, 0, 0)).get_width()

    # =========================
    # Largeurs adaptatives
    # =========================
    label_col_w = max(
        text_width("Commande") + 20,
        text_width(f"P{n}") + 20,
        70
    )

    prov_col_w = max(
        text_width("Provision") + 20,
        text_width(str(max(provisions))) + 20,
        80
    )

    remaining_w = W - margin_left - margin_right - label_col_w - prov_col_w
    cell_w = max(60, min(remaining_w // m, 120))

    total_rows = n + 2
    remaining_h = H - margin_top - margin_bottom
    cell_h = max(40, min(remaining_h // total_rows, 70))

    # =========================
    # Scroll
    # =========================
    x_offset = 0
    y_offset = 0
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_DOWN:
                    y_offset -= 30
                elif event.key == pygame.K_UP:
                    y_offset = min(y_offset + 30, 0)
                elif event.key == pygame.K_RIGHT:
                    x_offset -= 30
                elif event.key == pygame.K_LEFT:
                    x_offset = min(x_offset + 30, 0)

        screen.fill((15, 15, 30))

        x0 = margin_left + x_offset
        y0 = margin_top + y_offset

        # =========================
        # Titre
        # =========================
        screen.blit(
            title_font.render(titre, True, (255, 255, 0)),
            (x0, y0 - 60)
        )

        # =========================
        # Grille
        # =========================
        for i in range(n + 2):
            for j in range(m + 2):

                if j == 0:
                    cw = label_col_w
                    x = x0
                elif j == m + 1:
                    cw = prov_col_w
                    x = x0 + label_col_w + m * cell_w
                else:
                    cw = cell_w
                    x = x0 + label_col_w + (j - 1) * cell_w

                y = y0 + i * cell_h

                pygame.draw.rect(screen, (180, 180, 180), (x, y, cw, cell_h), 1)

                # =========================
                # Contenu
                # =========================
                text = None
                color = (255, 255, 255)
                align = "center"

                # En-têtes colonnes
                if i == 0 and 1 <= j <= m:
                    text = f"C{j}"
                    color = (200, 200, 255)

                # En-tête Provision
                elif i == 0 and j == m + 1:
                    text = "Provision"
                    color = (200, 200, 255)

                # En-têtes lignes
                elif j == 0 and 1 <= i <= n:
                    text = f"P{i}"
                    color = (200, 200, 255)

                # Commande (label)
                elif i == n + 1 and j == 0:
                    text = "Commande"
                    color = (200, 200, 255)
                    align = "left"

                # Provisions
                elif 1 <= i <= n and j == m + 1:
                    text = str(provisions[i - 1])

                # Commandes
                elif i == n + 1 and 1 <= j <= m:
                    text = str(commandes[j - 1])

                # Somme
                elif i == n + 1 and j == m + 1:
                    text = str(sum(commandes))

                # Coûts + valeurs
                elif 1 <= i <= n and 1 <= j <= m:
                    ci, cj = i - 1, j - 1

                    # Surbrillance Balas-Hammer
                    if highlight == (ci, cj):
                        pygame.draw.rect(
                            screen,
                            (255, 80, 80),
                            (x + 2, y + 2, cw - 4, cell_h - 4),
                            2
                        )

                    # coût (petit, fixe)
                    surf_cost = font_cost.render(
                        str(couts[ci][cj]), True, (150, 150, 255)
                    )
                    screen.blit(surf_cost, (x + 5, y + 4))

                    # Valeur par défaut
                    val = None

                    if valeurs is not None and ci < len(valeurs) and cj < len(valeurs[ci]):
                        val = valeurs[ci][cj]

                    if val is not None:
                        # couleur selon la valeur
                        if val == 0:
                            color_val = (120, 180, 255)  #  bleu pour 0
                        else:
                            color_val = (255, 120, 120)  #  rouge pour ≠ 0

                        surf_val = font_value.render(str(val), True, color_val)
                        screen.blit(
                            surf_val,
                            (
                                x + cw // 2 - surf_val.get_width() // 2,
                                y + cell_h - surf_val.get_height() - 4
                            )
                        )

                if text is not None:
                    surf = font_label.render(text, True, color)
                    if align == "left":
                        screen.blit(surf, (x + 6, y + cell_h // 2 - surf.get_height() // 2))
                    else:
                        screen.blit(
                            surf,
                            (
                                x + cw // 2 - surf.get_width() // 2,
                                y + cell_h // 2 - surf.get_height() // 2
                            )
                        )

        # Aide
        screen.blit(
            help_font.render(
                "Flèches : déplacer | ENTER / ESC : retour",
                True,
                (150, 150, 150)
            ),
            (20, H - 25)
        )

        pygame.display.flip()
        clock.tick(60)



def balas_hammer_pygame(screen, couts, steps):
    """
    Visualisation Pygame complète de la méthode de Balas-Hammer
    avec pénalités Δ lignes / Δ colonnes intégrées
    et cadrage ADAPTATIF comme afficher_matrice_transport_pygame.
    """

    index = 0
    total = len(steps)

    x_offset = 0
    y_offset = 0

    clock = pygame.time.Clock()
    running = True

    # =========================
    # Polices
    # =========================
    font_cost = pygame.font.SysFont("consolas", 18)
    font_value = pygame.font.SysFont("consolas", 18)
    font_label = pygame.font.SysFont("consolas", 18)
    title_font = pygame.font.SysFont("consolas", 26, bold=True)
    info_font = pygame.font.SysFont("consolas", 18)

    W, H = screen.get_width(), screen.get_height()

    def text_width(text):
        return font_label.render(text, True, (0, 0, 0)).get_width()

    # =========================
    # Boucle principale
    # =========================
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_n:
                    index = min(index + 1, total - 1)

                elif event.key == pygame.K_p:
                    index = max(index - 1, 0)

                elif event.key == pygame.K_l:
                    index = total - 1

                elif event.key == pygame.K_LEFT:
                    x_offset += 30
                elif event.key == pygame.K_RIGHT:
                    x_offset -= 30
                elif event.key == pygame.K_UP:
                    y_offset += 30
                elif event.key == pygame.K_DOWN:
                    y_offset -= 30

        # =========================
        # Données de l’étape
        # =========================
        step = steps[index]

        x = step["x"]
        offre = step["offre"]
        demande = step["demande"]
        pen_l = step.get("pen_l")
        pen_c = step.get("pen_c")
        choix = step.get("choix")

        n = len(x)
        m = len(x[0])

        # =========================
        # Cadrage adaptatif (IDENTIQUE logique afficher)
        # =========================
        margin_left = 40
        margin_top = 120
        margin_right = 40
        margin_bottom = 60

        label_col_w = max(
            text_width("Commande") + 20,
            text_width(f"P{n}") + 20,
            70
        )

        prov_col_w = max(
            text_width("Provision") + 20,
            text_width(str(max(offre))) + 20,
            80
        )

        delta_col_w = max(
            text_width(f"Δ{index+1} ligne") + 20,
            80
        )

        remaining_w = (
            W - margin_left - margin_right
            - label_col_w - prov_col_w - delta_col_w
        )

        cell_w = max(60, min(remaining_w // m, 120))

        total_rows = n + 3
        remaining_h = H - margin_top - margin_bottom
        cell_h = max(40, min(remaining_h // total_rows, 70))

        # =========================
        # Affichage
        # =========================
        screen.fill((15, 15, 30))

        x0 = margin_left + x_offset
        y0 = margin_top + y_offset

        # ---------- Titre ----------
        screen.blit(
            title_font.render(
                f"Balas-Hammer — Étape {index+1}/{total}",
                True,
                (255, 255, 0)
            ),
            (x0, y0 - 60)
        )

        # ---------- Grille ----------
        for i in range(n + 3):
            for j in range(m + 3):

                if j == 0:
                    cw = label_col_w
                    x_pos = x0
                elif j == m + 1:
                    cw = prov_col_w
                    x_pos = x0 + label_col_w + m * cell_w
                elif j == m + 2:
                    cw = delta_col_w
                    x_pos = x0 + label_col_w + m * cell_w + prov_col_w
                else:
                    cw = cell_w
                    x_pos = x0 + label_col_w + (j - 1) * cell_w

                y_pos = y0 + i * cell_h

                pygame.draw.rect(
                    screen, (180, 180, 180),
                    (x_pos, y_pos, cw, cell_h), 1
                )

                text = None
                color = (255, 255, 255)

                # ---------- En-têtes ----------
                if i == 0 and 1 <= j <= m:
                    text = f"C{j}"
                    color = (200, 200, 255)

                elif j == 0 and 1 <= i <= n:
                    text = f"P{i}"
                    color = (200, 200, 255)

                elif i == 0 and j == m + 1:
                    text = "Provision"
                    color = (200, 200, 255)

                elif i == 0 and j == m + 2:
                    text = f"Δ{index+1} ligne"
                    color = (255, 140, 140)

                elif i == n + 1 and j == 0:
                    text = "Commande"
                    color = (200, 200, 255)

                elif i == n + 2 and j == 0:
                    text = f"Δ{index+1} col"
                    color = (255, 140, 140)

                # ---------- Provisions ----------
                elif 1 <= i <= n and j == m + 1:
                    text = str(offre[i - 1])

                # ---------- Commandes ----------
                elif i == n + 1 and 1 <= j <= m:
                    text = str(demande[j - 1])

                elif i == n + 1 and j == m + 1:
                    text = str(sum(demande))

                # ---------- Δ lignes ----------
                elif 1 <= i <= n and j == m + 2 and pen_l:
                    if pen_l[i - 1] >= 0:
                        text = str(pen_l[i - 1])
                        color = (255, 120, 120)

                # ---------- Δ colonnes ----------
                elif i == n + 2 and 1 <= j <= m and pen_c:
                    if pen_c[j - 1] >= 0:
                        text = str(pen_c[j - 1])
                        color = (255, 120, 120)

                # ---------- Case transport ----------
                elif 1 <= i <= n and 1 <= j <= m:
                    ci, cj = i - 1, j - 1

                    if choix == (ci, cj):
                        pygame.draw.rect(
                            screen, (255, 80, 80),
                            (x_pos + 2, y_pos + 2, cw - 4, cell_h - 4), 2
                        )

                    screen.blit(
                        font_cost.render(str(couts[ci][cj]), True, (150, 150, 255)),
                        (x_pos + 5, y_pos + 4)
                    )

                    val = x[ci][cj]

                    # couleur selon valeur
                    if val == 0:
                        color_val = (120, 180, 255)  #
                    else:
                        color_val = (255, 120, 120)  #

                    surf_val = font_value.render(str(val), True, color_val)
                    screen.blit(
                        surf_val,
                        (
                            x_pos + cw // 2 - surf_val.get_width() // 2,
                            y_pos + cell_h - surf_val.get_height() - 4
                        )
                    )

                if text is not None:
                    surf = font_label.render(text, True, color)
                    screen.blit(
                        surf,
                        (x_pos + cw//2 - surf.get_width()//2,
                         y_pos + cell_h//2 - surf.get_height()//2)
                    )

        # ---------- Aide ----------
        screen.blit(
            info_font.render(
                "N/P : navigation | L : fin | Flèches : déplacer | ENTER / ESC : retour",
                True, (160, 160, 160)
            ),
            (20, H - 30)
        )

        pygame.display.flip()
        clock.tick(60)


def balas_hammer_steps(couts, offre, demande):
    """
    Génère les états successifs de la méthode de Balas-Hammer.
    Chaque état correspond à UNE itération logique.
    """

    couts = deepcopy(couts)
    offre = offre[:]
    demande = demande[:]

    n = len(offre)
    m = len(demande)

    x = [[0 for _ in range(m)] for _ in range(n)]
    lignes_actives = [True] * n
    colonnes_actives = [True] * m

    steps = []

    def calculer_penalites():
        pen_l = [-1] * n
        pen_c = [-1] * m

        for i in range(n):
            if not lignes_actives[i] or offre[i] == 0:
                continue
            costs = [couts[i][j] for j in range(m)
                     if colonnes_actives[j] and demande[j] > 0]
            if len(costs) >= 2:
                costs.sort()
                pen_l[i] = costs[1] - costs[0]
            elif len(costs) == 1:
                pen_l[i] = costs[0]

        for j in range(m):
            if not colonnes_actives[j] or demande[j] == 0:
                continue
            costs = [couts[i][j] for i in range(n)
                     if lignes_actives[i] and offre[i] > 0]
            if len(costs) >= 2:
                costs.sort()
                pen_c[j] = costs[1] - costs[0]
            elif len(costs) == 1:
                pen_c[j] = costs[0]

        return pen_l, pen_c

    iteration = 1

    while any(offre) and any(demande):
        pen_l, pen_c = calculer_penalites()
        max_l = max(pen_l)
        max_c = max(pen_c)
        delta_max = max(max_l, max_c)

        if delta_max < 0:
            break

        choisir_ligne = max_l >= max_c

        if choisir_ligne:
            idx_pen_ligne = pen_l.index(max_l)
            i = idx_pen_ligne
            j = min(
                (j for j in range(m) if colonnes_actives[j] and demande[j] > 0),
                key=lambda j: couts[i][j]
            )
            idx_pen_col = None
        else:
            idx_pen_col = pen_c.index(max_c)
            j = idx_pen_col
            i = min(
                (i for i in range(n) if lignes_actives[i] and offre[i] > 0),
                key=lambda i: couts[i][j]
            )
            idx_pen_ligne = None

        q = min(offre[i], demande[j])

        #  Sauvegarde ÉTAT AVANT allocation
        steps.append({
            "iteration": iteration,
            "x": deepcopy(x),
            "offre": offre[:],
            "demande": demande[:],
            "pen_l": pen_l[:],
            "pen_c": pen_c[:],
            "i_choisie": i,
            "j_choisie": j,
            "idx_pen_ligne": idx_pen_ligne,
            "idx_pen_col": idx_pen_col,
            "delta_max": delta_max,
            "choisir_ligne": choisir_ligne
        })

        # Allocation
        x[i][j] += q
        offre[i] -= q
        demande[j] -= q

        if offre[i] == 0:
            lignes_actives[i] = False
        if demande[j] == 0:
            colonnes_actives[j] = False

        iteration += 1

    # 🔹 État final
    steps.append({
        "iteration": iteration,
        "x": deepcopy(x),
        "offre": offre[:],
        "demande": demande[:],
        "pen_l": None,
        "pen_c": None,
        "i_choisie": None,
        "j_choisie": None,
        "idx_pen_ligne": None,
        "idx_pen_col": None,
        "delta_max": None,
        "choisir_ligne": None
    })

    return steps


def afficher_console_pygame(screen, texte, titre="Sortie console"):
    """
    Affiche un texte multi-lignes (console) dans Pygame avec scroll.
    """
    font = pygame.font.SysFont("consolas", 18)
    title_font = pygame.font.SysFont("consolas", 26, bold=True)

    lignes = texte.split("\n")

    x_offset = 0
    y_offset = 0

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_DOWN:
                    y_offset -= 25
                elif event.key == pygame.K_UP:
                    y_offset = min(y_offset + 25, 0)

        screen.fill((15, 15, 30))

        # Titre
        screen.blit(
            title_font.render(titre, True, (255, 255, 0)),
            (40, 20)
        )

        y = 80 + y_offset
        for ligne in lignes:
            surf = font.render(ligne, True, (200, 200, 200))
            screen.blit(surf, (40, y))
            y += 22

        # Aide
        help_font = pygame.font.SysFont("consolas", 18)
        screen.blit(
            help_font.render("↑ / ↓ : défiler | ENTER / ESC : retour", True, (150, 150, 150)),
            (40, screen.get_height() - 30)
        )

        pygame.display.flip()
        clock.tick(60)

def dessiner_graphe_basis_pygame(screen, basis, titre="Graphe de la base"):
    """
    Dessine le graphe biparti correspondant à la matrice basis en Pygame.
    Pi à gauche, Cj à droite.
    Une arête est tracée si basis[i][j] == True.
    """

    n = len(basis)
    m = len(basis[0])

    W, H = screen.get_width(), screen.get_height()

    font = pygame.font.SysFont("consolas", 20)
    title_font = pygame.font.SysFont("consolas", 26, bold=True)

    # ---------------------------
    # Positions des nœuds
    # ---------------------------
    margin_x = 120
    margin_y = 120

    left_x = margin_x
    right_x = W - margin_x

    # Espacement vertical
    spacing_P = (H - 2 * margin_y) // max(n, 1)
    spacing_C = (H - 2 * margin_y) // max(m, 1)

    pos_P = {}
    pos_C = {}

    for i in range(n):
        pos_P[i] = (left_x, margin_y + i * spacing_P)

    for j in range(m):
        pos_C[j] = (right_x, margin_y + j * spacing_C)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    running = False

        screen.fill((15, 15, 30))

        # ---------------------------
        # Titre
        # ---------------------------
        screen.blit(
            title_font.render(titre, True, (255, 255, 0)),
            (W // 2 - 200, 40)
        )

        # ---------------------------
        # Arêtes (avant les nœuds)
        # ---------------------------
        for i in range(n):
            for j in range(m):
                if basis[i][j]:
                    pygame.draw.line(
                        screen,
                        (200, 200, 200),
                        pos_P[i],
                        pos_C[j],
                        2
                    )

        # ---------------------------
        # Nœuds Pi
        # ---------------------------
        for i, (x, y) in pos_P.items():
            pygame.draw.circle(screen, (80, 160, 255), (x, y), 22)
            txt = font.render(f"P{i+1}", True, (0, 0, 0))
            screen.blit(
                txt,
                (x - txt.get_width() // 2, y - txt.get_height() // 2)
            )

        # ---------------------------
        # Nœuds Cj
        # ---------------------------
        for j, (x, y) in pos_C.items():
            pygame.draw.circle(screen, (160, 220, 160), (x, y), 22)
            txt = font.render(f"C{j+1}", True, (0, 0, 0))
            screen.blit(
                txt,
                (x - txt.get_width() // 2, y - txt.get_height() // 2)
            )

        # ---------------------------
        # Aide
        # ---------------------------
        help_font = pygame.font.SysFont("consolas", 18)
        screen.blit(
            help_font.render("ENTER / ESC : retour", True, (150, 150, 150)),
            (40, H - 30)
        )

        pygame.display.flip()
        clock.tick(60)


def trace_execution_transport(
    groupe,
    equipe,
    numero_probleme,
    methode,              # "no" ou "bh"
    couts,
    provisions,
    commandes,
    trace_logs            # liste de chaînes déjà produites (console)
):
    """
    Écrit la trace complète d'un problème de transport dans un fichier texte.
    """

    # ========= 1) Dossier =========
    dossier = "Traces_Transport"
    os.makedirs(dossier, exist_ok=True)

    # ========= 2) Nom du fichier =========
    filename = f"{groupe}-{equipe}-trace{numero_probleme}-{methode}.txt"
    filepath = os.path.join(dossier, filename)

    # ========= 3) Construction du texte =========
    with open(filepath, "a", encoding="utf-8") as f:

        # --- EN-TÊTE ---
        f.write("===== TRACE D'EXECUTION - PROBLEME DE TRANSPORT =====\n\n")
        f.write(f"Groupe        : {groupe}\n")
        f.write(f"Equipe        : {equipe}\n")
        f.write(f"Problème n°   : {numero_probleme}\n")
        f.write(f"Méthode       : {methode.upper()}\n\n")

        # --- DONNÉES INITIALES ---
        f.write("===== DONNEES INITIALES =====\n\n")
        f.write("Matrice des coûts :\n")
        f.write(afficher_matrice_text(couts) + "\n")

        f.write("Provisions :\n")
        f.write(str(provisions) + "\n\n")

        f.write("Commandes :\n")
        f.write(str(commandes) + "\n\n")

        # --- TRACE DE LA RESOLUTION ---
        f.write("===== RESOLUTION =====\n\n")

        for bloc in trace_logs:
            f.write(bloc + "\n")
            f.write("-" * 60 + "\n")

    print(f"Trace enregistrée dans : {filepath}")

def afficher_matrice_text(couts, valeurs=None, provisions=None, commandes=None):
    import io
    buffer = io.StringIO()

    n = len(couts)
    m = len(couts[0])

    contenus = []
    contenus += [f"C{j+1}" for j in range(m)]
    if provisions is not None:
        contenus.append("Provisions")

    contenus += [f"P{i+1}" for i in range(n)]
    for row in couts:
        contenus += [str(c) for c in row]

    max_len = max(len(x) for x in contenus)
    col = max_len + 2

    def ligne_sep():
        return "+" + "+".join("-" * col for _ in range(m + (1 if provisions else 0) + 1)) + "+"

    def write_row(cells):
        buffer.write("|" + "|".join(str(c).center(col) for c in cells) + "|\n")

    buffer.write(ligne_sep() + "\n")
    header = [""] + [f"C{j+1}" for j in range(m)]
    if provisions:
        header.append("Prov")
    write_row(header)
    buffer.write(ligne_sep() + "\n")

    for i in range(n):
        row = [f"P{i+1}"] + [str(couts[i][j]) for j in range(m)]
        if provisions:
            row.append(str(provisions[i]))
        write_row(row)

        if valeurs:
            write_row([""] + [str(valeurs[i][j]) for j in range(m)] + ([""] if provisions else []))

        buffer.write(ligne_sep() + "\n")

    if commandes:
        write_row(["Cmd"] + [str(c) for c in commandes] + [str(sum(commandes))])

    return buffer.getvalue()

def executer_transport_complet(
    screen,
    methode,          # "no" ou "bh"
    couts,
    provisions,
    commandes,
    numero_probleme,
    groupe,
    equipe
):
    """
    Exécute :
    - Nord-Ouest ou Balas
    - Marche-pied
    - Coût de transport
    - Traces texte
    - Lancement visualisation Pygame
    """

    trace_logs = []

    # =========================
    # SOLUTION INITIALE
    # =========================
    if methode == "no":
        valeurs, basis = nord_ouest(provisions, commandes)
        texte_no = print_f(nord_ouest, provisions, commandes)
        trace_logs.append("===== NORD-OUEST =====\n" + texte_no)

    elif methode == "bh":
        valeurs, basis = methode_balas_hammer(couts, provisions, commandes)
        texte_bh = print_f(methode_balas_hammer, couts, provisions, commandes)
        trace_logs.append("===== BALAS-HAMMER =====\n" + texte_bh)

    # =========================
    # MARCHE-PIED (CALCUL)
    # =========================
    texte_mp = print_f(marche_pied, valeurs, basis, couts)
    trace_logs.append("===== MARCHE-PIED =====\n" + texte_mp)

    val_opt, basis_opt = marche_pied(valeurs, basis, couts)

    # =========================
    # COÛT DE TRANSPORT
    # =========================
    texte_cout = print_f(
        calculer_cout_transport,
        val_opt,
        couts,
        True
    )
    trace_logs.append("===== COUT OPTIMAL =====\n" + texte_cout)
    cout_total = calculer_cout_transport(
        val_opt,
        couts,
        afficher=False
    )

    # =========================
    # TRACE FICHIER
    # =========================
    trace_execution_transport(
        groupe=groupe,
        equipe=equipe,
        numero_probleme=numero_probleme,
        methode=methode,
        couts=couts,
        provisions=provisions,
        commandes=commandes,
        trace_logs=trace_logs
    )

    # =========================
    # AFFICHAGE CONSOLE PYGAME
    # =========================



def marche_pied_pygame(screen, x_init, basis_init, cost):
    """
    Visualisation pédagogique COMPLETE du marche-pied
    - un seul écran
    - graphe + matrice transport (coûts+quantités)
    - E + c* + Δ + cycle
    - navigation: N/P (étapes), O/K (itérations), M/L (début/fin)
    """


    # =========================================================
    # OUTILS (petits helpers internes)
    # =========================================================
    def compute_prov_cmd(x):
        n = len(x)
        m = len(x[0])
        prov = [sum(x[i][j] for j in range(m)) for i in range(n)]
        cmd = [sum(x[i][j] for i in range(n)) for j in range(m)]
        return prov, cmd

    # =========================================================
    # 1) PRÉ-CALCUL de toutes les itérations + sous-étapes
    # =========================================================
    x = deepcopy(x_init)
    basis = deepcopy(basis_init)

    states = []
    iteration = 1

    while True:
        basis_prev = deepcopy(basis)
        x_before = deepcopy(x)

        # Graphe + tests
        graph = build_graph(x, basis)
        connexe, visited = is_connected(graph)
        acyclique = is_acyclic(graph)

        added_repair = None
        if (not connexe) or (not acyclique):
            basis, added_repair = repair_degenerate_base(x, basis, cost, graph, visited)
            graph = build_graph(x, basis)
            connexe, visited = is_connected(graph)
            acyclique = is_acyclic(graph)

        # Potentiels + c* + delta
        E = compute_potentials(x, cost, basis)
        c_star = compute_potential_costs(x, E)
        delta = compute_reduced_costs(cost, c_star)
        entering = find_entering_arc(x, basis, delta)

        # arêtes ajoutées (par rapport à la base précédente)
        added_edges = []
        for i in range(len(basis)):
            for j in range(len(basis[0])):
                if basis[i][j] and not basis_prev[i][j]:
                    added_edges.append((i, j))

        # Cycle + theta + mise à jour (si entering)
        cycle = None
        theta = None
        x_after = None
        basis_after = None

        if entering is not None:
            cycle = build_cycle_for_entering_arc(x, basis, entering)

            # theta = min sur les '-' du cycle
            th = 10**18
            for (i, j), sign in cycle:
                if sign == '-':
                    th = min(th, x[i][j])
            theta = th

            # simuler la maj (sans casser x avant)
            x2 = deepcopy(x)
            for (i, j), sign in cycle:
                if sign == '+':
                    x2[i][j] += theta
                else:
                    x2[i][j] -= theta

            b2 = deepcopy(basis)
            ei, ej = entering
            b2[ei][ej] = True

            x_after = x2
            basis_after = b2

            # appliquer réellement pour itération suivante
            x = x2
            basis = b2

        prov_before, cmd_before = compute_prov_cmd(x_before)
        if x_after is not None:
            prov_after, cmd_after = compute_prov_cmd(x_after)
        else:
            prov_after, cmd_after = prov_before, cmd_before

        states.append({
            "iteration": iteration,
            "basis_prev": basis_prev,
            "basis": deepcopy(basis),
            "added": added_edges,
            "added_repair": added_repair,

            "connexe": connexe,
            "acyclique": acyclique,

            "x_before": x_before,
            "prov_before": prov_before,
            "cmd_before": cmd_before,

            "E": deepcopy(E),
            "c_star": deepcopy(c_star),
            "delta": deepcopy(delta),
            "entering": entering,

            "cycle": deepcopy(cycle),
            "theta": theta,

            "x_after": deepcopy(x_after),
            "prov_after": prov_after,
            "cmd_after": cmd_after,
            "basis_after": deepcopy(basis_after),
        })

        if entering is None:
            break

        iteration += 1

        # =========================================================
        # 2) PYGAME: navigation
        # =========================================================
        index = 0  # itération affichée
        step = 0  # sous-étape affichée (0..4)
        total = len(states)
        x_offset = 0
        y_offset = 0
        y_offset = 0

        MAX_SCROLL = 0
        MIN_SCROLL = -1600  # ajuste si besoin

        clock = pygame.time.Clock()
        running = True

        font = pygame.font.SysFont("consolas", 16)
        font_small = pygame.font.SysFont("consolas", 14)
        title_font = pygame.font.SysFont("consolas", 14, bold=True)

        W, H = screen.get_width(), screen.get_height()
        # =========================
        # LAYOUT FIXE
        # =========================
        GRAPH_TOP = 90  # début graphe
        GRAPH_HEIGHT = 260  # hauteur max du graphe (FIXE)
        GRAPH_BOTTOM = GRAPH_TOP + GRAPH_HEIGHT

        CONTENT_TOP = GRAPH_BOTTOM + 20  # début zone scrollable

        # couleurs des arêtes ajoutées (une couleur par itération)
        COLORS = [
            (255, 80, 80),  # rouge
            (80, 255, 80),  # vert
            (80, 80, 255),  # bleu
            (255, 255, 80),  # jaune
            (255, 80, 255),  # magenta
            (80, 255, 255),  # cyan
            (255, 160, 80),  # orange
            (180, 120, 255),  # violet
        ]

        # =========================================================
        # 3) AFFICHAGE UNIQUE (graphe + tables)
        # =========================================================
        def draw_text(x, y, txt, col=(200, 200, 200), f=None):
            if f is None:
                f = font
            screen.blit(f.render(txt, True, col), (x, y))

        def draw_table(top_left_x, top_left_y, title, data, row_labels=None, col_labels=None,
                       cell_w=56, cell_h=26,
                       value_color_fn=None,
                       highlight_cells=None,
                       border_col=(180, 180, 180),
                       title_col=(255, 255, 0)):
            """
            Dessine une table simple.
            data: liste de listes
            highlight_cells: dict {(i,j): color} (contour)
            value_color_fn: (i,j,val) -> couleur texte
            """
            if highlight_cells is None:
                highlight_cells = {}

            n = len(data)
            m = len(data[0]) if n else 0

            y = top_left_y
            draw_text(top_left_x, y, title, title_col, f=font)
            y += 22

            x0 = top_left_x
            y0 = y

            label_w = 50 if row_labels else 0

            # en-têtes colonnes
            if col_labels:
                for j in range(m):
                    cx = x0 + label_w + j * cell_w
                    pygame.draw.rect(screen, border_col, (cx, y0, cell_w, cell_h), 1)
                    t = font_small.render(str(col_labels[j]), True, (200, 200, 255))
                    screen.blit(t, (cx + cell_w // 2 - t.get_width() // 2, y0 + cell_h // 2 - t.get_height() // 2))
                y0 += cell_h

            # contenu
            for i in range(n):
                # label ligne
                if row_labels:
                    cx = x0
                    pygame.draw.rect(screen, border_col, (cx, y0, label_w, cell_h), 1)
                    t = font_small.render(str(row_labels[i]), True, (200, 200, 255))
                    screen.blit(t, (cx + label_w // 2 - t.get_width() // 2, y0 + cell_h // 2 - t.get_height() // 2))

                for j in range(m):
                    cx = x0 + label_w + j * cell_w
                    pygame.draw.rect(screen, border_col, (cx, y0, cell_w, cell_h), 1)

                    # contour highlight
                    if (i, j) in highlight_cells:
                        pygame.draw.rect(screen, highlight_cells[(i, j)], (cx + 2, y0 + 2, cell_w - 4, cell_h - 4), 2)

                    val = data[i][j]
                    col = (230, 230, 230)
                    if value_color_fn:
                        col = value_color_fn(i, j, val)

                    t = font_small.render(str(val), True, col)
                    screen.blit(t, (cx + cell_w // 2 - t.get_width() // 2, y0 + cell_h // 2 - t.get_height() // 2))

                y0 += cell_h

            return y0 + 10  # position y après table

        def draw_transport_matrix(x, cost, prov, cmd, top_left_x, top_left_y, title,
                                  highlight=None, highlight_cells=None):

            """
            Une matrice "transport" (coût petit en haut à gauche, quantité en bas)
            """
            n = len(cost)
            m = len(cost[0])

            # dimensions adaptatives simplifiées
            margin_right = 30
            margin_bottom = 30
            label_w = 55
            prov_w = 80
            cell_w = max(60, min((W - top_left_x - margin_right - label_w - prov_w) // m, 110))
            cell_h = 52

            draw_text(top_left_x, top_left_y, title, (255, 255, 0), f=font)
            y0 = top_left_y + 22
            x0 = top_left_x

            if highlight_cells is None:
                highlight_cells = {}

            # En-têtes colonnes
            for j in range(m):
                cx = x0 + label_w + j * cell_w
                pygame.draw.rect(screen, (180, 180, 180), (cx, y0, cell_w, cell_h), 1)
                draw_text(cx + cell_w // 2 - 12, y0 + cell_h // 2 - 10, f"C{j}", (200, 200, 255), f=font_small)

            # "Provision"
            cx = x0 + label_w + m * cell_w
            pygame.draw.rect(screen, (180, 180, 180), (cx, y0, prov_w, cell_h), 1)
            draw_text(cx + 10, y0 + cell_h // 2 - 10, "Prov", (200, 200, 255), f=font_small)

            # Lignes
            for i in range(n):
                ry = y0 + (i + 1) * cell_h

                # label ligne
                pygame.draw.rect(screen, (180, 180, 180), (x0, ry, label_w, cell_h), 1)
                draw_text(x0 + 10, ry + cell_h // 2 - 10, f"P{i}", (200, 200, 255), f=font_small)

                # cellules coût/quantité
                for j in range(m):
                    cx = x0 + label_w + j * cell_w
                    pygame.draw.rect(screen, (180, 180, 180), (cx, ry, cell_w, cell_h), 1)

                    if highlight == (i, j):
                        pygame.draw.rect(screen, (255, 80, 80), (cx + 2, ry + 2, cell_w - 4, cell_h - 4), 2)

                    # coût (petit)
                    draw_text(cx + 4, ry + 4, str(cost[i][j]), (150, 150, 255), f=font_small)

                    # quantité (couleur: 0 bleu, sinon rouge)
                    q = x[i][j]
                    q_col = (120, 180, 255) if q == 0 else (255, 120, 120)
                    t = font.render(str(q), True, q_col)
                    screen.blit(t, (cx + cell_w // 2 - t.get_width() // 2, ry + cell_h - t.get_height() - 4))

                # provision
                cx = x0 + label_w + m * cell_w
                pygame.draw.rect(screen, (180, 180, 180), (cx, ry, prov_w, cell_h), 1)
                draw_text(cx + 10, ry + cell_h // 2 - 10, str(prov[i]), (230, 230, 230), f=font_small)

            # ligne commandes
            ry = y0 + (n + 1) * cell_h
            pygame.draw.rect(screen, (180, 180, 180), (x0, ry, label_w, cell_h), 1)
            draw_text(x0 + 4, ry + cell_h // 2 - 10, "Cmd", (200, 200, 255), f=font_small)

            for j in range(m):
                cx = x0 + label_w + j * cell_w
                pygame.draw.rect(screen, (180, 180, 180), (cx, ry, cell_w, cell_h), 1)
                draw_text(cx + 10, ry + cell_h // 2 - 10, str(cmd[j]), (230, 230, 230), f=font_small)

            # somme
            cx = x0 + label_w + m * cell_w
            pygame.draw.rect(screen, (180, 180, 180), (cx, ry, prov_w, cell_h), 1)
            draw_text(cx + 10, ry + cell_h // 2 - 10, str(sum(cmd)), (230, 230, 230), f=font_small)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        running = False

                    elif event.key == pygame.K_n:
                        step = min(step + 1, 4)
                    elif event.key == pygame.K_p:
                        step = max(step - 1, 0)

                    elif event.key == pygame.K_k:
                        index = min(index + 1, total - 1)
                        step = 0
                    elif event.key == pygame.K_o:
                        index = max(index - 1, 0)
                        step = 0

                    elif event.key == pygame.K_m:
                        index = 0
                        step = 0
                    elif event.key == pygame.K_l:
                        index = total - 1
                        step = 0

                    elif event.key == pygame.K_DOWN:
                        y_offset = max(y_offset - 30, MIN_SCROLL)

                    elif event.key == pygame.K_UP:
                        y_offset = min(y_offset + 30, MAX_SCROLL)

                    #x0 = margin_left + x_offset
                    #y0 = margin_top + y_offset

            screen.fill((15, 15, 30))
            st = states[index]

            # =========================
            # TITRE
            # =========================
            screen.blit(
                title_font.render(
                    f"Marche-Pied — Itération {st['iteration']}  ({index + 1}/{total})   Étape {step}/4",
                    True, (255, 255, 0)
                ),
                (40, 20)
            )

            if index == total - 1:
                val_opt, basis_opt = marche_pied(x_init, basis, cost)

                cout_total = calculer_cout_transport(
                    val_opt,
                    cost,
                    afficher=False
                )
                draw_text(
                    40, 50,
                    "Solution optimale atteinte",
                    (120, 255, 120),
                    f=font
                )
                draw_text(
                    40, 75,
                    f"Coût total optimal : {cout_total}",
                    (255, 200, 80),
                    f=font
                )

            # =========================
            # ZONE GAUCHE: Graphe biparti
            # =========================
            # =========================
            # ZONE GRAPHE (FIXE)
            # =========================
            basis_prev = st["basis_prev"]
            added = st["added"]

            n = len(basis_prev)
            m = len(basis_prev[0])

            graph_width = 420
            center_x = W // 2
            left_x = center_x - graph_width // 2
            right_x = center_x + graph_width // 2

            top = GRAPH_TOP
            usable_h = GRAPH_HEIGHT - 40

            spacing_P = max(30, usable_h // max(n, 1))
            spacing_C = max(30, usable_h // max(m, 1))

            pos_P = {i: (left_x, top + 20 + i * spacing_P) for i in range(n)}
            pos_C = {j: (right_x, top + 20 + j * spacing_C) for j in range(m)}

            NODE_R = 14

            # Arêtes anciennes
            for i in range(n):
                for j in range(m):
                    if basis_prev[i][j]:
                        pygame.draw.line(screen, (120, 120, 120), pos_P[i], pos_C[j], 1)

            # Arêtes ajoutées
            add_col = COLORS[index % len(COLORS)]
            for (i, j) in added:
                pygame.draw.line(screen, add_col, pos_P[i], pos_C[j], 2)

            # Nœuds P
            for i, (x0, y0) in pos_P.items():
                pygame.draw.circle(screen, (80, 160, 255), (x0, y0), NODE_R)
                t = font_small.render(f"P{i}", True, (0, 0, 0))
                screen.blit(t, (x0 - t.get_width() // 2, y0 - t.get_height() // 2))

            # Nœuds C
            for j, (x0, y0) in pos_C.items():
                pygame.draw.circle(screen, (160, 220, 160), (x0, y0), NODE_R)
                t = font_small.render(f"C{j}", True, (0, 0, 0))
                screen.blit(t, (x0 - t.get_width() // 2, y0 - t.get_height() // 2))

            # =========================
            # INFOS (toujours visibles)
            # =========================
            yinfo = 90
            draw_text(40, yinfo, f"Connexe : {'OUI' if st['connexe'] else 'NON'}", (200, 200, 200))
            yinfo += 22
            draw_text(40, yinfo, f"Acyclique : {'OUI' if st['acyclique'] else 'NON'}", (200, 200, 200))
            yinfo += 22
            draw_text(40, yinfo, f"Arête entrante : {st['entering']}", (255, 140, 140))
            yinfo += 22

            if st["added_repair"] is not None:
                draw_text(40, yinfo, f"Réparation ajoutée : {st['added_repair']}", (255, 200, 120))
                yinfo += 22

            # =========================
            # ZONE DROITE: tables selon step
            # step 0: juste graphe + E
            # step 1: matrice transport AVANT
            # step 2: c*
            # step 3: delta + entering
            # step 4: cycle + theta + matrice transport APRES
            # =========================
            right_panel_x = 40
            right_panel_y = H - 270

            if step == 0:
                y = CONTENT_TOP + y_offset
                draw_text(40, y, "Potentiels E :", (200, 200, 255))
                y += 20
                for k in sorted(st["E"].keys()):
                    draw_text(40, y, f"{k} = {st['E'][k]}", (230, 230, 230))
                    y += 18



            elif step == 1:
                draw_transport_matrix(
                    st["x_before"], cost,
                    st["prov_before"], st["cmd_before"],
                    40, CONTENT_TOP + y_offset,
                    title="Matrice transport (avant)"
                )


            elif step == 2:
                draw_table(
                    40, CONTENT_TOP + y_offset,
                    "Coûts potentiels c*",
                    st["c_star"],
                    row_labels=[f"P{i}" for i in range(len(st["c_star"]))],
                    col_labels=[f"C{j}" for j in range(len(st["c_star"][0]))]
                )


            elif step == 3:

                draw_table(

                    40, CONTENT_TOP + y_offset,

                    "Coûts marginaux Δ = c - c*",

                    st["delta"],

                    row_labels=[f"P{i}" for i in range(len(st["delta"]))],

                    col_labels=[f"C{j}" for j in range(len(st["delta"][0]))],

                    highlight_cells={st["entering"]: (255, 80, 80)} if st["entering"] else None,

                    value_color_fn=lambda i, j, v: (120, 180, 255) if v == 0 else (255, 120, 120) if v < 0 else (
                    230, 230, 230)

                )



            elif step == 4 and st["cycle"]:

                y = CONTENT_TOP + y_offset

                draw_text(40, y, f"Cycle (θ = {st['theta']}) :", (255, 255, 0))

                y += 20

                for (i, j), sgn in st["cycle"]:
                    draw_text(40, y, f"({i},{j}) {sgn}", (80, 255, 80) if sgn == '+' else (255, 80, 80))

                    y += 18

                y += 10

                draw_transport_matrix(

                    st["x_before"], cost,

                    st["prov_before"], st["cmd_before"],

                    380, y,

                    title="Transport (avant) + cycle"

                )

                y += 260

                draw_transport_matrix(

                    st["x_after"], cost,

                    st["prov_after"], st["cmd_after"],

                    380, y,

                    title="Transport (après)"

                )

            # =========================
            # AIDE
            # =========================
            draw_text(
                W - 820, 20,
                "↑↓ Scroll | N/P Étapes | O/K Itérations | M/L Début/Fin | ESC Retour",
                (150, 150, 150), font_small
            )
            draw_text(40, CONTENT_TOP - 30, f"y_offset = {y_offset}", (255, 80, 80))




            pygame.display.flip()
            clock.tick(60)



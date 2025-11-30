import pygame
import time
import random
import io, contextlib
import re, pygame  # Importation du module pour manipuler les expressions régulières
import sys, os #pour intérargir avec syst d'exploitation - manipuler les répertoires et fichiers
from fonction import *


# Initialisation de Pygame
pygame.init()

# Paramètres de la fenêtre
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Projet Recherche Opérationnelle - Résolution des problèmes de transport")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # dossier temporaire PyInstaller
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

backgrounds = [
    pygame.image.load(resource_path("images/background1.png"))
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
    text = render_text_multiline(help_text, font, state.text_color, WIDTH - 40)

    y_offset = HEIGHT // 4  # Position de départ
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
        text = f"Entrez un numéro entre 0 et {total-1}:"
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
    sub_options = ["Afficher le matrice", "floydifier le matrice", "Aide",
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

from complexite import *


def main():
    # Valeurs de n imposées par l'énoncé
    n_values = [10, 40, 100, 400, 1000, 4000, 10000]

    # ⚠ Pour les gros n, 100 répétitions risque d'être très long.
    # Tu peux commencer par 5 ou 10 pour tester, puis passer à 100.
    nb_repetitions = 100

    print("=======================================")
    print("   PROJET TRANSPORT — ÉTUDE COMPLEXITÉ")
    print("=======================================")

    resultats = lancer_experiences(
        n_values=n_values,
        nb_repetitions=nb_repetitions
    )

    # Ajout des combinaisons θ + t
    ajouter_combinaisons(resultats)

    # Sauvegarde des résultats dans des fichiers CSV
    sauvegarder_resultats_bruts(resultats, "resultats_bruts.csv")
    sauvegarder_maximums(resultats, "resultats_max.csv")


    # Nuages de points
    tracer_nuage_points(resultats, "theta_NO", "Nuage de points θ_NO(n)")
    tracer_nuage_points(resultats, "theta_BH", "Nuage de points θ_BH(n)")
    tracer_nuage_points(resultats, "t_NO", "Nuage de points t_NO(n)")
    tracer_nuage_points(resultats, "t_BH", "Nuage de points t_BH(n)")
    tracer_nuage_points(resultats, "theta_plus_t_NO", "Nuage de points θ_NO + t_NO")
    tracer_nuage_points(resultats, "theta_plus_t_BH", "Nuage de points θ_BH + t_BH")

    # Pire des cas (enveloppe supérieure = max)
    max_theta_NO = calculer_max_par_n(resultats, "theta_NO")
    max_theta_BH = calculer_max_par_n(resultats, "theta_BH")
    max_t_NO = calculer_max_par_n(resultats, "t_NO")
    max_t_BH = calculer_max_par_n(resultats, "t_BH")
    max_theta_plus_t_NO = calculer_max_par_n(resultats, "theta_plus_t_NO")
    max_theta_plus_t_BH = calculer_max_par_n(resultats, "theta_plus_t_BH")

    tracer_courbe_max(max_theta_NO, "Pire cas θ_NO(n)")
    tracer_courbe_max(max_theta_BH, "Pire cas θ_BH(n)")
    tracer_courbe_max(max_t_NO, "Pire cas t_NO(n)")
    tracer_courbe_max(max_t_BH, "Pire cas t_BH(n)")
    tracer_courbe_max(max_theta_plus_t_NO, "Pire cas θ_NO + t_NO")
    tracer_courbe_max(max_theta_plus_t_BH, "Pire cas θ_BH + t_BH")

    # Comparaison des deux algorithmes
    comp = calculer_rapport_NO_sur_BH(resultats)
    tracer_rapports_max(comp["rapports_max"])

    print("\n🔥 Étude de complexité terminée.")


if __name__ == "__main__":
    main()
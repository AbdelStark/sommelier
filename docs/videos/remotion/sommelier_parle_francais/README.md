# Sommelier parle français. La souveraineté se mesure.

Vidéo promotionnelle du [run français de Sommelier](../../../results/french-run.md),
construite avec [Remotion](https://remotion.dev). 1920x1080, 30 images/seconde,
2 848 images (~95 s), rythmée pour que chaque carte, chaque barre et chaque
chiffre puissent réellement être lus. Silencieuse par choix : ajoutez une
musique dans votre logiciel de montage si la plateforme cible s’y prête.

## Le script, scène par scène

Les textes ci-dessous sont ceux qui apparaissent à l’écran, surtitre compris
(à la typographie près : l’écran utilise des espaces insécables). S’y ajoutent
seulement les libellés des barres (« anglais », « français », « base »,
« affiné »). Les horodatages tiennent compte des huit fondus de 14 images.

### Scène 1 · La question (0:00 à 0:10)

L’accroche. Deux constats posés l’un après l’autre, puis la question qui
lance la vidéo.

> SOMMELIER · LE RUN FRANÇAIS
>
> Vos clients parlent français.
>
> Vos agents raisonnent en anglais.
>
> **Combien ça coûte ?**
>
> Alors on a mesuré.

### Scène 2 · Le constat (0:10 à 0:20)

Le prix de la langue, mesuré sur le modèle de base. Deux barres, un badge.

> MÊME MODÈLE, MÊME TÂCHE, DEUX LANGUES
>
> En français, le même modèle perd 4,2 points.
>
> Appel complet exact : anglais 70,5 %, français 66,3 %, badge -4,2 pts.
>
> Une taxe silencieuse, sur chaque requête.

Note d’écran : appel complet exact · modèle de base · n=1000 en / n=879 fr ·
décodage déterministe · nvidia/Llama-3.1-Nemotron-Nano-8B-v1.

### Scène 3 · La méthode (0:20 à 0:33)

Le contrat de traduction, montré plutôt qu’expliqué. Deux terminaux côte à
côte tapent la même question, en anglais à gauche, en français à droite. La
réponse gold qui s’affiche sous les deux est identique.

> LA MÉTHODE · LE CONTRAT DE TRADUCTION
>
> Une seule variable : la langue de la requête.
>
> REQUÊTE · ANGLAIS
>
> user › What is the weather in Paris today?
>
> REQUÊTE · FRANÇAIS
>
> user › Quel temps fait-il à Paris aujourd’hui ?
>
> gold › {"name": "lookup_weather", "arguments": {"city": "Paris"}}
>
> Schémas et réponses gold : identiques à l’octet près ✓

Note d’écran : traducteur ouvert épinglé · valeurs protégées · 12,1 % de
paires écartées, et comptées.

### Scène 4 · Le résultat (0:33 à 0:49)

Deux temps. D’abord les barres du test français, modèle de base contre
adaptateur. Ensuite le parcours de l’écart de langue sur les trois modèles,
avec la ligne v2 mise en avant.

> LE RÉSULTAT
>
> Le test français, avant / après.
>
> JSON valide : 90,4 % → 99,5 % (+9,1 pts)
>
> F1 des arguments : 70,9 % → 92,1 % (+21,2 pts)
>
> Appel complet exact : 66,3 % → 87,3 % (+20,9 pts)
>
> Et face à l’anglais ?
>
> Modèle de base : -4,2 pts · Adaptateur v1 · anglais seul : -2,3 pts ·
> Adaptateur v2 · bilingue : +0,3 pt, TRANCHES MARGINALES
>
> Les tranches complètes sont presque alignées. **Descriptif, pas apparié.**

Note d’écran : run nemotron-8b-fr-full-4 · mêmes prompts et même parseur
pour les deux modèles · échec de parsing = échec.

### Scène 5 · Les moyens (0:49 à 0:56)

L’économie du run, en très grand.

> UNE L40S
>
> 7 HEURES
>
> **≈ 16 $**
>
> Ce run bilingue tient dans le prix estimé d’un déjeuner.

Note d’écran : 5 h 42 d’entraînement · pic mémoire 26 369 MiB · estimation
au tarif du run v1 (~8 $ / ~3,5 h).

### Scène 6 · La pile ouverte (0:56 à 1:06)

Ce qui rend le run possible, et reproductible par n’importe quelle équipe.

> LA PILE OUVERTE NVIDIA
>
> Rien de tout ça n’exige un labo.
>
> 01 MODÈLES OUVERTS · Nemotron Nano 8B : les poids, téléchargeables sur Hugging Face
>
> 02 LOGICIELS OUVERTS · NeMo Curator : curation de données sur GPU, licence Apache-2.0
>
> 03 RECETTES OUVERTES · Le pipeline de données Nemotron-CC, publié de bout en bout
>
> Les pelles sont à louer. **Le modèle, lui, est à vous.**

### Scène 7 · Le data flywheel (1:06 à 1:17)

La roue tourne à l’écran : cinq étapes s’allument à mesure que l’anneau se
remplit, puis la boucle se referme au centre.

> LE DATA FLYWHEEL, EN VRAI
>
> DONNÉES V1 → TRADUCTION SOUS CONTRAT → ENTRAÎNEMENT → ÉVALUATION →
> ARTEFACTS V2 → tour suivant ↻
>
> Chaque tour produit les données du suivant.

Note d’écran : le v2 a tourné sur les artefacts publiés du v1 · le tour
suivant peut partir d’ici.

### Scène 8 · La souveraineté (1:17 à 1:27)

La thèse, au présent et sans conditionnel.

> Personne ne peut en changer le prix.
>
> Personne ne peut le retirer.
>
> Personne ne peut vous en couper l’accès.
>
> **Vos poids. Vos données. Votre langue.**

Note d’écran : La souveraineté se mesure : tranches marginales à +0,3 pt
(fr n=879, en n=1000 ; estimation appariée non disponible).

### Scène 9 · L’appel (1:27 à 1:35)

> 🍷 SOMMELIER
>
> Code ouvert. Poids ouverts. Données ouvertes. Preuves ouvertes.
>
> github.com/AbdelStark/sommelier · hf.co/spaces/abdelstark/sommelier
>
> Licence MIT · métriques publiques, rapports vérifiables

## Politique de vérification des faits

Chaque affirmation à l’écran porte sa source et son statut de preuve :

- Les métriques, les écarts de langue, les effectifs (n=1000 en, n=879 fr),
  la durée d’entraînement, le pic mémoire et l’identité du run viennent de
  `docs/results/french-run.md` (run `nemotron-8b-fr-full-4`). Voir
  `src/data/facts.ts` pour la source de chaque chiffre.
- Les « 7 heures » sont la somme des étapes du pipeline consignées dans
  `docs/results/french-run.md` (25 220 s ≈ 7,0 h).
- L’estimation « ≈ 16 $ » applique aux 7 heures de ce run le tarif
  reconstruit du run v1 : ~8 $ (billet v1,
  `docs/blogposts/sommelier_blog_post.md`) pour ~3,5 h de pipeline (somme
  des étapes de `docs/results/reference-run.md`), soit 8 / 3,55 × 7,0 ≈
  15,8 $. Modal n’a pas exposé la facturation au run lui-même et le rapport
  enregistre le coût comme indisponible.
- Les affirmations sur la pile NVIDIA (NeMo Curator sous Apache-2.0,
  recette Nemotron-CC publique, poids Nemotron sous NVIDIA Open Model
  License) ont été vérifiées pour la première vidéo du dépôt contre GitHub,
  Hugging Face et le blog NVIDIA.
- L’exemple des deux terminaux illustre le contrat de traduction décrit
  dans `docs/concepts/data.md` : seule la requête est traduite, la réponse
  gold reste identique à l’octet près. La requête reprend l’exemple de
  service du README du dépôt.

## Utilisation

```bash
npm install
npm run dev      # Remotion Studio (aperçu interactif)
npm run render   # rend out/sommelier_parle_francais.mp4
```

Le mp4 rendu et `node_modules` sont ignorés par git ; seul le source est
versionné.

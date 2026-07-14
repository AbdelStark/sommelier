# Sommelier v2 : réduire l’écart français

*Sur les tranches historiques complètes, un petit modèle ouvert affichait quatre points de moins en français. Sept heures de GPU plus tard, les valeurs marginales française et anglaise ne diffèrent plus que de 0,3 point. Les métriques et durées remontent aux rapports publics ; le coût reste une extrapolation explicitement étiquetée. Récit du run, de la méthode, et de l’enjeu de souveraineté qui va avec.*

---

Le billet sur le [run v1](sommelier_blog_post.md) se refermait sur une promesse : la prochaine étape serait une tranche d’évaluation en français, parce qu’un appel d’outil devrait fonctionner aussi bien en français qu’en anglais, et que c’est le genre d’affirmation qu’on mesure au lieu de la supposer.

Promesse tenue. Et le sujet n’a rien d’académique. L’utilisateur simulé de Sommelier est une entreprise française. Ses clients écrivent en français. Son agent doit donc choisir le bon outil et remplir les bons arguments à partir de requêtes en français, des milliers de fois par jour. Si le modèle se dégrade dans la langue de vos utilisateurs, vous payez une taxe silencieuse sur chaque requête, et personne ne vous en communique le montant.

Alors je l’ai mesurée.

## La fin, d’abord

Comme pour le v1, les chiffres avant l’histoire. La métrique la plus dure du pipeline est l’appel complet exact : bonne fonction et bons arguments, jugés par correspondance exacte, échec de parsing compté comme échec. La voici, français moins anglais, pour les trois modèles que le projet a mesurés sur les mêmes tranches de test :

| Modèle | Anglais | Français | Écart |
|--------|---------|----------|-------|
| Modèle de base | 70,5 % | 66,3 % | -4,2 pts |
| Adaptateur v1, entraîné en anglais seul | 87,4 % | 85,1 % | -2,3 pts |
| Adaptateur v2, entraîné en anglais et en français | 87,0 % | **87,3 %** | **+0,3 pt** |

Trois constats descriptifs, dans l’ordre. Sur les tranches complètes, le taux français du modèle de base est inférieur d’environ quatre points au taux anglais. Le fine-tuning en anglais seul transfère l’essentiel de ses gains à la tranche française sans avoir vu une seule ligne de français, et la différence marginale passe à 2,3 points. Avec les données françaises, les valeurs marginales des cinq métriques se tiennent à moins d’un tiers de point. Les cohortes comptent toutefois 879 lignes françaises et 1 000 anglaises : l’artefact v2 ne permet ni d’appeler cette différence du « bruit », ni d’en faire un effet causal apparié.

Sur la tranche française seule, le gain brut entre le modèle de base et l’adaptateur v2 :

| Métrique | Base | Adaptateur v2 | Gain |
|----------|------|---------------|------|
| JSON valide | 90,4 % | **99,5 %** | +9,1 pts |
| Bonne fonction | 89,8 % | **99,0 %** | +9,2 pts |
| Arguments exacts | 66,6 % | **87,6 %** | +21,1 pts |
| F1 des arguments | 70,9 % | **92,1 %** | +21,2 pts |
| Appel complet exact | 66,3 % | **87,3 %** | +20,9 pts |

*Les gains sont calculés sur les valeurs non arrondies de la page de résultats, d’où le dixième d’écart possible avec une soustraction des pourcentages affichés.*

Le protocole tient en une phrase : 879 prompts français jamais vus à l’entraînement, décodage déterministe, parseur conservateur identique pour les deux modèles, et un rapport de comparaison qui refuse d’exister si les empreintes de configuration, de découpe et de prompts ne correspondent pas. Le run s’appelle `nemotron-8b-fr-full-4`, exécuté le 6 juillet 2026 sur une seule L40S. [La page de résultats](../results/french-run.md) donne les numérateurs, les dénominateurs et toutes les empreintes.

## Une seule variable

La règle du run : ne changer qu’une chose. Même modèle de base, mêmes hyperparamètres, même seed, même pipeline que le run de référence. La seule nouveauté est un jeu de données : une variante française appariée de chaque exemple retenu.

C’est là que l’expérience se joue, parce que traduire un jeu de données d’appels d’outils est un piège. Si la traduction touche au schéma de l’outil ou à la réponse attendue, les deux langues ne mesurent plus la même tâche et la comparaison ne vaut rien. D’où un contrat, appliqué par le code plutôt que par la bonne volonté :

- Seule la requête de l’utilisateur est traduite. Les schémas d’outils et les réponses gold restent identiques à l’octet près, ce que la préparation vérifie ligne par ligne.
- Les valeurs que la réponse attend (nombres, identifiants, symboles) sont des spans protégés : le traducteur doit les restituer tels quels, sinon la paire est écartée.
- Chaque paire écartée est comptée, motif par motif, dans un résumé publié avec le jeu de données. Rien ne disparaît en silence.
- Chaque ligne française hérite de la découpe train, validation ou test de sa source anglaise. Une traduction d’un exemple d’entraînement ne peut pas contaminer le jeu de test.

Le traducteur est lui-même un modèle ouvert figé à une version précise, Mistral-Nemo-Instruct-2407 en décodage déterministe, exécuté sur Modal et consigné dans le manifeste du run. Pas d’API propriétaire dans la boucle : la traduction est aussi rejouable que le reste du pipeline.

Mon bug préféré du run est très français. Le traducteur, consciencieux, écrivait les nombres à la française : « 2,5 » là où la réponse gold exige « 2.5 ». La virgule décimale cassait les spans protégés et envoyait à la poubelle des paires parfaitement bonnes. La normalisation est depuis déterministe et testée (`normalize_numeric_spans`), et la virgule a retrouvé sa place : dans la prose, pas dans les arguments JSON.

Au total, 14 936 paires françaises publiées, dont 13 113 pour l’entraînement. Sur la tranche de test, 12,1 % des paires sont écartées parce que leur réponse gold recopie de l’anglais depuis la requête : les traduire aurait rompu le contrat. Ce chiffre est une limite assumée de la tranche française, pas une note honteuse en bas de page : il figure dans la carte du jeu de données, avec le décompte par motif.

## Ce que l’anglais y perd

L’honnêteté d’abord : la tranche anglaise du v2 se situe 0,3 à 0,8 point sous le v1 selon la métrique (87,0 contre 87,4 en appel complet exact). À n=1000, c’est à moins d’une erreur-type. Impossible de trancher, avec un seul seed, entre du bruit et un petit prix payé pour la couverture bilingue, et le rapport le dit dans ces termes, parce qu’un chiffre qui n’avoue pas son incertitude est une publicité.

La comparaison, elle, est solidement ancrée : l’empreinte du jeu de prompts anglais du v2 est identique octet pour octet à celle du run de référence. Les 1 000 prompts anglais qui donnent le 87,0 % sont exactement ceux qui donnaient le 87,4 % du v1.

## Une affaire de souveraineté

Venons-en à la raison d’être de tout ceci. En 2026, une entreprise française qui construit un produit agentique sur un modèle fermé accepte trois dépendances sur lesquelles elle n’a aucune prise. Le fournisseur peut changer ses prix. Il peut retirer le modèle : onze modèles Claude ont été retirés entre juillet 2025 et juin 2026. Il peut conditionner l’accès : vérification d’identité gouvernementale pour certains modèles d’OpenAI, licence Llama 4 qui exclut les entités domiciliées dans l’Union européenne, modèle de pointe d’Anthropic réservé aux organisations approuvées. Toutes ces conditions d’accès sont publiées, et elles ont été vérifiées une à une pour [la vidéo du projet](https://github.com/AbdelStark/sommelier/tree/main/docs/videos/remotion/sommelier_is_here_sovereign_ai_is_needed).

Il existe une quatrième dépendance, dont on parle moins : la langue. Un modèle construit et évalué d’abord en anglais fonctionne moins bien en français. L’intuition est désormais un chiffre : quatre points d’appel complet exact en moins sur le modèle de base. Si votre produit vit en français, cette dégradation est structurelle, silencieuse, et vous n’avez aucun levier dessus tant que le modèle appartient à quelqu’un d’autre.

La souveraineté, ramenée à la pratique, c’est ce levier. Les poids chez vous. Les transformations de données déclarées et comptées. L’évaluation rejouable depuis les empreintes. Et votre langue traitée comme un cas de premier ordre, pas comme un marché secondaire qu’on couvrira plus tard.

Ce run montre à quoi ressemble le levier une fois actionné. La différence marginale a été mesurée, attaquée avec 13 113 lignes d’entraînement traduites sous contrat, puis réduite à 0,3 point entre les tranches complètes ; l’estimation appariée reste volontairement non revendiquée. Le coût de l’opération tient sur une ligne : sept heures d’une L40S, pic mémoire à 26 369 MiB, entraînement de 5 h 42. Modal n’a pas exposé de données de facturation au run lui-même, le rapport enregistre donc le coût comme indisponible, mais au tarif observé du run v1 (environ huit dollars pour trois heures et demie de GPU, la somme des étapes du run de référence), on parle d’une extrapolation d’environ seize dollars.

Pour la France en particulier, l’enjeu dépasse le confort linguistique. Les secteurs qui ont le plus à attendre des agents, l’administration, la santé, la banque, le droit, sont aussi ceux qui peuvent le moins externaliser leurs données et leurs dépendances. Un modèle de 8 milliards de paramètres, affiné sur vos schémas d’outils et vos requêtes en français, qui tourne sur un GPU loué à l’heure ou possédé en propre, est une réponse concrète à cette équation. Une réponse étroite, cantonnée à la tâche qui fait tenir les agents debout : l’appel d’outil.

## Le data flywheel, en vrai

« Data flywheel » est d’ordinaire un mot de slide : la boucle où l’usage produit des données, où les données améliorent le modèle, où le modèle améliore le produit, et où le produit relance l’usage. NVIDIA en a fait un pilier de sa vision des agents. Ce run en est un tour complet, avec les mains dans la mécanique :

1. Le run v1 a publié ses artefacts : les découpes exactes du jeu de données anglais, l’adaptateur, les rapports d’évaluation.
2. Le run v2 a consommé ces artefacts comme matière première. Chaque ligne française dérive d’une ligne anglaise publiée et porte l’identifiant de sa source.
3. Il publie à son tour un jeu de données français avec sa carte de provenance, un adaptateur bilingue et des rapports verrouillés par empreintes.
4. Le tour suivant peut partir de là demain matin. Rien dans la boucle n’appartient à un tiers.

La roue ne tourne que parce que chaque étape est déclarée. Des données dont on ignore les filtres sont un point de départ inutilisable. Des paires écartées sans compteur sont un biais invisible. Une évaluation sans empreintes est un chiffre d’ambiance. La leçon d’ingénierie du flywheel : rendez chaque tour auditable, sinon le tour suivant hérite de vos approximations et les amplifie.

## La pile NVIDIA, en pratique

Ce que cette expérience doit à la pile ouverte de NVIDIA, concrètement :

- **Le modèle de base.** Llama-3.1-Nemotron-Nano-8B-v1, poids téléchargeables, sous NVIDIA Open Model License avec les obligations Llama 3.1, consignées dans le dépôt et vérifiées avant chaque publication. Un 8B s’affine en QLoRA sur un seul GPU de 44 Gio.
- **Les logiciels.** NeMo Curator, l’outillage de curation de données de la suite NeMo, est sous Apache-2.0.
- **Les recettes.** Le pipeline de données Nemotron-CC est publié de bout en bout. Pas un article qui décrit le pipeline : le pipeline.

La lecture stratégique mérite une phrase, parce qu’elle concerne directement les entreprises françaises. NVIDIA gagne à ce que mille équipes fassent ce genre de run, puisqu’elles consomment du calcul accéléré. Les équipes y gagnent des modèles qui leur appartiennent. L’alignement est d’une netteté inhabituelle, et il est assumé publiquement des deux côtés. Pour une entreprise française, la conclusion opérationnelle est simple : les pelles sont à louer, le modèle reste à vous.

## Les limites, sans détour

- Les requêtes françaises sont des traductions machine, relues par échantillonnage. Ce ne sont pas des requêtes écrites par des francophones.
- La tranche française exclut les 12,1 % de paires dont les arguments gold contiennent de l’anglais recopié depuis la requête. Elle penche donc légèrement vers les exemples aux arguments neutres.
- Le prompt système reste en anglais dans les deux langues, par choix de conception : l’effet de la langue d’instruction n’est pas mesuré.
- Un run, un seed, deux langues. Rien ici ne classe le modèle sur un benchmark public.

## Artefacts

- Les résultats complets, avec empreintes : [la page du run français](../results/french-run.md)
- L’adaptateur bilingue et ses rapports : [huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora](https://huggingface.co/abdelstark/llama-3.1-nemotron-nano-8b-xlam-tool-calling-fr-en-lora)
- Le jeu de données français, avec provenance et décomptes : [huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits-fr](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits-fr)
- Les découpes anglaises, inchangées depuis le v1 : [huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits](https://huggingface.co/datasets/abdelstark/sommelier-xlam-single-call-splits)
- Le code, la reproduction, la démo : [github.com/AbdelStark/sommelier](https://github.com/AbdelStark/sommelier)

*Sommelier est sous licence MIT. L’adaptateur est un dérivé de Llama 3.1 (« Built with Llama ») ; les obligations de licence du modèle et des jeux de données sont consignées dans le dépôt et vérifiées par un contrôle automatique avant chaque publication.*

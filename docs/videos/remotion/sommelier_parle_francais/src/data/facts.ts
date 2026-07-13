// Chaque chiffre de ce fichier remonte à un artefact du dépôt. Ne rien
// modifier sans revérifier la source.
//
// Source principale : docs/results/french-run.md (run nemotron-8b-fr-full-4,
// exécuté le 2026-07-06 sur une L40S, tranches de test n=1000 en / n=879 fr,
// décodage déterministe, parseur conservateur, échec de parsing = échec).

// Affichage à la française : virgule décimale, espace insécable avant %.
export const fr = (n: number, digits = 1): string =>
  n.toFixed(digits).replace('.', ',');

export const pct = (n: number, digits = 1): string => `${fr(n, digits)} %`;

export type Metric = {
  label: string;
  base: number; // pourcentage
  adapter: number; // pourcentage
  delta: string; // delta du rapport, pas la soustraction des valeurs arrondies
};

// Source : docs/results/french-run.md, tableau « French slice (n=879) ».
// JSON valide 0,9044 -> 0,9954 (+0,0910) ; F1 des arguments 0,7091 -> 0,9208
// (+0,2117) ; appel complet exact 0,6633 -> 0,8726 (+0,2093).
export const METRIQUES_FR: Metric[] = [
  {label: 'JSON valide', base: 90.4, adapter: 99.5, delta: '+9,1 pts'},
  {label: 'F1 des arguments', base: 70.9, adapter: 92.1, delta: '+21,2 pts'},
  {label: 'Appel complet exact', base: 66.3, adapter: 87.3, delta: '+20,9 pts'},
];

// Source : docs/results/french-run.md, tableau « The language gap, measured
// three ways » (appel complet exact, modèle de base : en 0,7050 / fr 0,6633).
export const CONSTAT = {
  en: 70.5,
  frSlice: 66.3,
  ecart: '-4,2 pts',
};

// Source : même tableau. Base -0,0417 ; adaptateur v1 (jalon M1, issue #108)
// -0,0230 ; adaptateur v2 +0,0026.
export type EtapeEcart = {
  modele: string;
  ecart: string;
  referme?: boolean;
};

export const ECARTS: EtapeEcart[] = [
  {modele: 'Modèle de base', ecart: '-4,2 pts'},
  {modele: 'Adaptateur v1 · anglais seul', ecart: '-2,3 pts'},
  {modele: 'Adaptateur v2 · bilingue', ecart: '+0,3 pt', referme: true},
];

// Source : docs/results/french-run.md « Runtime and cost ». Somme des étapes
// du pipeline : 25 220 s ≈ 7 h. Entraînement 20 540 s = 5 h 42.
// L'estimation « ≈ 16 $ » applique aux 7 h de ce run le tarif reconstruit du
// run v1 : ~8 $ (docs/blogposts/sommelier_blog_post.md, « about eight
// dollars ») pour 12 766,7 s ≈ 3,55 h de pipeline (somme des étapes de
// docs/results/reference-run.md « Runtime and cost »), soit
// 8 / 3,55 × 7,0 ≈ 15,8 $. Le rapport du run enregistre le coût facturé
// comme indisponible.
export const MOYENS = {
  gpu: 'UNE L40S',
  temps: '7 HEURES',
  prix: '≈ 16 $',
};

export const MOYENS_NOTE =
  '5 h 42 d’entraînement · pic mémoire 26 369 MiB · estimation au tarif du run v1 (~8 $ / ~3,5 h)';

// Illustration du contrat de traduction, calquée sur l'exemple de service du
// README : la requête française est la traduction fidèle de la même question,
// et la réponse gold reste identique à l'octet près (docs/concepts/data.md).
export const DEMO_EN = 'What is the weather in Paris today?';
export const DEMO_FR = 'Quel temps fait-il à Paris aujourd’hui ?';
export const DEMO_GOLD =
  '{"name": "lookup_weather",\n  "arguments": {"city": "Paris"}}';

// Source : docs/results/french-run.md « Claim boundaries » (traducteur
// Mistral-Nemo-Instruct-2407 épinglé, spans protégés, 12,1 % de paires
// écartées car la réponse gold recopie de l'anglais depuis la requête).
// Le taux vaut pour la tranche de test (1000 -> 879) et, par coïncidence
// arrondie, pour l'ensemble publié (17 000 -> 14 936, soit 12,1 %).
export const CONTRAT_NOTE =
  'traducteur ouvert épinglé · valeurs protégées · 12,1 % de paires écartées, et comptées';

export const MODELE_LIGNE =
  'nvidia/Llama-3.1-Nemotron-Nano-8B-v1 · QLoRA · 15 000 en + 13 113 fr';

export const EVAL_NOTE =
  'appel complet exact · modèle de base · n=1000 en / n=879 fr · décodage déterministe';

export const RESULTATS_NOTE =
  'run nemotron-8b-fr-full-4 · mêmes prompts et même parseur pour les deux modèles · échec de parsing = échec';

// Source : docs/results/french-run.md (le run consomme les découpes
// anglaises publiées du v1 et publie le jeu de données fr + l'adaptateur
// bilingue + les rapports verrouillés par empreintes).
export const FLYWHEEL_ETAPES = [
  'DONNÉES V1',
  'TRADUCTION SOUS CONTRAT',
  'ENTRAÎNEMENT',
  'ÉVALUATION',
  'ARTEFACTS V2',
];

export const SOUVERAINETE_NOTE =
  'La souveraineté se mesure : écart français refermé, +0,3 pt (n=879).';

# Maze — expérience « latent disentangle » (h1/h2)

## Hypothèse
Le co-train hiérarchique Level 2 (latent partagé) **régressait** vs Level 1 (46.9 % < 65.6 %).
On teste : *la régression vient du **latent partagé**, pas de la hiérarchie.* Un seul latent
ne peut pas servir à la fois la **géométrie** (distance) et le **contrôle** (action) sans se polluer.

## Idée
Split du latent `z` (sur la dim canal) en deux blocs décorrélés :
- **h1 = `z[:, :k]`** → porte la **géométrie** : la perte de distance temporelle (`tdist`) n'agit que sur h1.
- **h2 = `z[:, k:]`** → porte le **contrôle** : l'IDM (inverse dynamics) ne pousse son gradient que dans h2
  (on `.detach()` h1 dans le chemin IDM, donc le module IDM garde sa dim — aucun changement d'archi).
- **décorrélation croisée** h1 ⟂ h2 (cross-corrélation² → 0).

Objectif : que `‖E(s₀) − E(s)‖` sur **h1** reflète la distance *atteignable* (nb de pas), et fasse
apparaître une **crête au niveau des portes** du labyrinthe (au lieu d'un dégradé radial qui ignore les murs).

## Ce qui a changé dans le code (rétro-compatible, défaut = off)
- `eb_jepa/losses.py` → `VC_IDM_Sim_Regularizer` : 2 params `dist_split_frac`, `cross_cov_coeff`
  + routage tdist→h1, IDM→h2 (stop-grad h1), méthode `_cross_decorr`.
- `examples/ac_video_jepa/main.py` : pass-through des 2 params depuis la config.
- `examples/ac_video_jepa/viz_distance_landscape.py` : option `--dims h1` (restreint la distance au bloc géométrie).
- Config : `cfgs/train/maze/train_maze_disentangle.yaml` = copie de `train_maze_temporal.yaml`
  **+ `dist_split_frac: 0.5`, `cross_cov_coeff: 1.0`** (seule différence → isole 1 variable).

## Lire le résultat
```bash
# bloc géométrie h1 du run disentangle
python -m examples.ac_video_jepa.viz_distance_landscape --model_folder <run_disentangle> --dims h1 --out land_split_h1.png
# baseline latent partagé (train_maze_temporal) pour comparer
python -m examples.ac_video_jepa.viz_distance_landscape --model_folder <run_temporal>   --out land_shared.png
```
**Succès** = `land_split_h1.png` montre une crête de distance aux portes (deux côtés d'un mur = loin),
là où le latent partagé donne un dégradé radial. Surveiller aussi `cross_cov_loss` (doit chuter).

## Limites
- Hyperparams `dist_split_frac=0.5` / `cross_cov_coeff=1.0` = points de départ, à ajuster.
- Le split ne s'applique qu'au chemin **non-projeté** (`use_proj: false`, `idm_after_proj: false`).
- Approche complémentaire (non incluse) pour aller plus loin : tête quasi-métrique + négatifs
  cross-trajectoire (pour l'explosion sur états non-atteignables) au lieu de la L2 brute.

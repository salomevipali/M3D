# Avatar Pipeline

Pipeline **Photo → Avatar 3D GLB + Métriques corporelles**

---

## Structure

```
avatar_project/
│
├── steps/                          ← modules unitaires (1 responsabilité chacun)
│   ├── __init__.py
│   ├── step1_skeleton_extract.py   ← MediaPipe → skeleton.json
│   ├── step2_smpl_fit.py           ← skeleton.json → betas SMPL-X
│   └── step3_export_glb.py         ← betas → avatar.glb
│
├── pipeline/                       ← orchestration + métriques
│   ├── __init__.py
│   ├── run.py                      ← point d'entrée principal ✅
│   └── metrics.py                  ← calcul body fat, BMR, mensurations…
│
├── output/                         ← fichiers générés (gitignore)
│   ├── avatar.glb
│   └── metrics.json
│
├── smpl_models/                    ← modèles SMPL-X (non inclus, à télécharger)
│   └── smplx/
│       ├── SMPLX_FEMALE.npz
│       └── SMPLX_MALE.npz
│
├── data/                           ← vos photos de test
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Modèles SMPL-X (obligatoire)
Téléchargez les modèles sur https://smpl-x.is.tue.mpg.de/ (gratuit, inscription requise)  
Placez `SMPLX_FEMALE.npz` et `SMPLX_MALE.npz` dans `smpl_models/smplx/`

---

## Usage

### Ligne de commande

```bash
python -m pipeline.run \
    --front  data/photo_face.jpg \
    --side   data/photo_profil.jpg \
    --height 170 \
    --weight 65 \
    --age    28 \
    --sex    female \
    --output output/ \
    --pose   tpose
```

**Résultat :**
```
output/
├── avatar.glb      ← mesh 3D lisible sur mobile (Three.js, SceneKit, etc.)
└── metrics.json    ← body fat, BMR, mensurations, morphotype…
```

### Programmatique (intégration future API)

```python
from pipeline.run import run_pipeline

result = run_pipeline(
    image_front = "data/face.jpg",
    image_side  = "data/side.jpg",
    height_cm   = 170,
    weight_kg   = 65,
    age         = 28,
    sex         = "female",
    output_dir  = "output/",
    pose        = "tpose",   # ou "relaxed"
)

print(result["glb_path"])      # chemin absolu du .glb
print(result["metrics"])       # dict body fat, BMR, etc.
```

---

## Poses disponibles

| Nom       | Description                         |
|-----------|-------------------------------------|
| `tpose`   | T-pose (défaut, idéale pour rigging)|
| `relaxed` | Bras légèrement baissés, naturel    |

---

## Sorties

### `avatar.glb`
Mesh SMPL-X texturé, format **glTF Binary**.  
Compatible : Three.js · React Three Fiber · SceneKit (iOS) · SceneView (Android) · Babylon.js

### `metrics.json`
```json
{
  "body_fat_percent": 22.5,
  "fat_mass_kg": 14.6,
  "lean_mass_kg": 50.4,
  "muscle_mass_kg": 25.2,
  "bone_mass_kg": 2.5,
  "bmr_kcal": 1487,
  "visceral_fat_index": 4.2,
  "whr": 0.79,
  "body_type": "mésomorphe",
  "confidence_score": 0.65,
  "measurements": {
    "waist_cm": 72.0,
    "hip_cm": 91.0,
    "chest_cm": 88.0,
    "neck_cm": 34.0,
    "thigh_cm": 55.0,
    "shoulder_width_cm": 40.0,
    "height_cm": 170.0,
    "bmi": 22.5
  }
}
```
"# M3D" 

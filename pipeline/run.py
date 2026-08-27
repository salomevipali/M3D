"""
pipeline/run.py
===============

Pipeline complet :

    photo face
        +
    photo profil
        +
    données utilisateur
        ↓
    MediaPipe + segmentation
        ↓
    fitting SMPL-X
        ↓
    métriques
        ↓
    avatar GLB
"""

import argparse
import json
import os
import sys

import numpy as np


# =============================================================================
# EVITE LE CONFLIT OPENMP WINDOWS
# =============================================================================

os.environ.setdefault(
    "KMP_DUPLICATE_LIB_OK",
    "TRUE",
)


# =============================================================================
# RACINE PROJET
# =============================================================================

_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if _ROOT not in sys.path:

    sys.path.insert(
        0,
        _ROOT,
    )


# =============================================================================
# IMPORTS
# =============================================================================

from steps.step1_skeleton_extract import (
    extract_skeleton,
)

from steps.step2_smpl_fit import (
    fit_smpl,
)

from steps.step3_export_glb import (
    export_avatar,
)

from pipeline.metrics import (
    compute_metrics,
)


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(

    image_front: str,

    image_side: str,

    height_cm: float,

    weight_kg: float,

    age: int,

    sex: str,
    activity_level: int,

    output_dir: str = "output",

    pose: str = "tpose",

) -> dict:

    """
    Pipeline complet :
        2 photos → GLB + métriques
    """

    # =========================================================================
    # OUTPUT
    # =========================================================================

    if not os.path.isabs(
        output_dir
    ):

        output_dir = os.path.join(
            _ROOT,
            output_dir,
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print(
        f"  [pipeline] "
        f"output_dir résolu → "
        f"{output_dir}"
    )

    # =========================================================================
    # CHEMINS
    # =========================================================================

    skeleton_face_path = os.path.join(

        output_dir,

        "skeleton_face.json",

    )

    skeleton_side_path = os.path.join(

        output_dir,

        "skeleton_side.json",

    )

    glb_path = os.path.join(

        output_dir,

        "avatar.glb",

    )

    metrics_path = os.path.join(

        output_dir,

        "metrics.json",

    )

    # =========================================================================
    # ETAPE 1 — FACE
    # =========================================================================

    _header(

        "1/3",

        "Extraction squelette — photo face",

    )

    extract_skeleton(

        image_front,

        skeleton_face_path,

    )

    # =========================================================================
    # ETAPE 1 — PROFIL
    # =========================================================================

    _header(

        "—",

        "Extraction squelette — photo profil",

    )

    extract_skeleton(

        image_side,

        skeleton_side_path,

    )

    # =========================================================================
    # CHARGEMENT SQUELETTES
    # =========================================================================

    with open(

        skeleton_face_path,

        "r",

        encoding="utf-8",

    ) as f:

        skeleton_face = json.load(
            f
        )

    with open(

        skeleton_side_path,

        "r",

        encoding="utf-8",

    ) as f:

        skeleton_side = json.load(
            f
        )

    # =========================================================================
    # ETAPE 2 — SMPL-X
    # =========================================================================

    _header(

        "2/3",

        "Fitting SMPL-X",

    )

    params = fit_smpl(

        skeleton_path=
            skeleton_face_path,

        height_cm=
            height_cm,

        sex=
            sex,

        skeleton_side_path=
            skeleton_side_path,

    )

    # =========================================================================
    # GENERATION MESH
    # =========================================================================

    _header(

        "—",

        "Calcul des métriques corporelles",

    )

    import torch
    import smplx

    model = smplx.create(

        params["model_path"],

        model_type="smplx",

        gender=sex,

        use_pca=False,

        batch_size=1,

    )

    betas_t = (

        torch.tensor(
            params["betas"]
        )

        .float()

        .unsqueeze(0)

    )

    with torch.no_grad():

        mesh_out = model(

            betas=betas_t,

            return_verts=True,

        )

    # =========================================================================
    # VERTICES
    # =========================================================================

    vertices = (

        mesh_out

        .vertices

        .detach()

        .cpu()

        .numpy()[0]

        *

        params[
            "scale_factor"
        ]

    )

    faces = model.faces

    # =========================================================================
    # JOINTS SMPL-X
    # =========================================================================

    joints = None

    if hasattr(

        mesh_out,

        "joints",

    ):

        joints = (

            mesh_out

            .joints

            .detach()

            .cpu()

            .numpy()[0]

            *

            params[
                "scale_factor"
            ]

        )

    # =========================================================================
    # METRIQUES
    # =========================================================================
    #
    # IMPORTANT :
    #
    # La formule ellipse est dans metrics.py.
    #
    # run.py transmet seulement les deux skeletons.
    #
    # waist_cm reste calculé par metrics.py exactement comme avant.
    # =========================================================================

    metrics = compute_metrics(

        vertices=
            vertices,

        faces=
            faces,

        height_cm=
            height_cm,

        weight_kg=
            weight_kg,

        age=
            age,

        sex=
            sex,

        joints=
            joints,

        skeleton_face=
            skeleton_face,

        skeleton_side=
            skeleton_side,
            
        activity_level=activity_level,

    )

    # =========================================================================
    # SAUVEGARDE
    # =========================================================================

    with open(

        metrics_path,

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            metrics,

            f,

            indent=2,

            ensure_ascii=False,

        )

    print(

        f"  [pipeline] "
        f"metrics.json écrit → "
        f"{metrics_path}"

    )

    # =========================================================================
    # ETAPE 3 — GLB
    # =========================================================================

    _header(

        "3/3",

        "Export avatar GLB",

    )

    glb_ok = True

    try:

        export_avatar(

            params,

            glb_path,

            pose=pose,

        )

    except Exception as e:

        glb_ok = False

        print(

            f"  [pipeline] "
            f"⚠ Export GLB échoué "
            f"({e}) "
            f"— metrics.json reste disponible"

        )

    # =========================================================================
    # RESUME
    # =========================================================================

    _summary(

        glb_path
        if glb_ok
        else None,

        metrics_path,

        metrics,

        output_dir,

    )

    return {

        "glb_path":

            (

                os.path.abspath(
                    glb_path
                )

                if glb_ok

                else None

            ),

        "metrics_path":

            os.path.abspath(
                metrics_path
            ),

        "metrics":

            metrics,

    }


# =============================================================================
# AFFICHAGE
# =============================================================================

def _header(

    step: str,

    title: str,

):

    print(
        f"\n{'─' * 56}"
    )

    print(
        f"  [{step}] {title}"
    )

    print(
        f"{'─' * 56}"
    )


def _summary(

    glb_path,

    metrics_path,

    metrics,

    output_dir,

):

    m = metrics.get(

        "measurements",

        {},

    )

    print(
        f"\n{'═' * 56}"
    )

    print(
        "  ✅ PIPELINE TERMINÉ"
    )

    print(
        f"{'═' * 56}"
    )

    print(

        f"  📁 Dossier : "
        f"{output_dir}/"

    )

    glb_label = (

        os.path.basename(
            glb_path
        )

        if glb_path

        else

        "❌ échec export"

    )

    print(

        f"  🌐 Avatar GLB : "
        f"{glb_label}"

    )

    print(

        f"  📊 Métriques : "
        f"{os.path.basename(metrics_path)}"

    )

    print(
        f"{'─' * 56}"
    )


    print(

        f"  Masse grasse : "
        f"{metrics.get('fat_mass_kg', '?')} kg"

    )


    print(

        f"  Masse musculaire : "
        f"{metrics.get('muscle_mass_kg', '?')} kg"

    )


    print(

        f"  BMR : "
        f"{metrics.get('bmr_kcal', '?')} kcal/j"

    )

    if m:

        print(
            f"{'─' * 56}"
        )

        for key, label in [
            (
                "belly_cm",
                "Tour de ventre      ",
            ),

            (
                "shoulder_circumference_cm",
                "Tour épaules        ",
            ),

        ]:

            val = m.get(
                key
            )

            if (
                val is not None
                and
                val > 0
            ):

                print(

                    f"  {label}: "
                    f"{val:.1f} cm"

                )

    print(
        f"{'═' * 56}\n"
    )


# =============================================================================
# CLI
# =============================================================================

def main():

    run_pipeline(

        image_front=
            "C:/Users/Restart/Desktop/"
            "VIPAVATAR-VIPANALYSE/"
            "avatar_project_local/"
            "data/morgan_face.jpeg",
        image_side=
            "C:/Users/Restart/Desktop/"
            "VIPAVATAR-VIPANALYSE/"
            "avatar_project_local/"
            "data/morgan_profil.jpeg",

        height_cm=185,

        weight_kg=84.7,

        age=42,

        sex="male",
        

        output_dir="output",

        pose="relaxed",

        activity_level = 5,

    )


if __name__ == "__main__":

    main()
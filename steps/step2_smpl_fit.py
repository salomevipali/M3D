"""
STEP 2 — Fitting SMPL-X
=======================

Phase 1 :
    fitting de la morphologie depuis la photo de face

    - joints MediaPipe
    - largeur d'épaules
    - silhouette réelle autour des épaules
    - profil vertical de largeur

Phase 2 :
    correction de profondeur depuis la photo de profil

Sortie :
    {
        betas,
        scale_factor,
        height_cm,
        sex,
        model_path,
        shoulder_y_norm
    }
"""

import json
import os

import numpy as np
from scipy.optimize import minimize


_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


SMPL_MODELS = {

    "female":
        os.path.join(
            _ROOT,
            "smpl_models",
            "smplx",
            "SMPLX_FEMALE.npz",
        ),

    "male":
        os.path.join(
            _ROOT,
            "smpl_models",
            "smplx",
            "SMPLX_MALE.npz",
        ),

    "neutral":
        os.path.join(
            _ROOT,
            "smpl_models",
            "smplx",
            "SMPLX_NEUTRAL.npz",
        ),
}


# =============================================================================
# CORRESPONDANCES MEDIAPIPE → SMPL-X
# =============================================================================

MP_TO_SMPLX = [

    (11, 16, 2.0),
    (12, 17, 2.0),

    (13, 18, 1.5),
    (14, 19, 1.5),

    (23, 1, 5.0),
    (24, 2, 5.0),

    (25, 4, 2.0),
    (26, 5, 2.0),

    (27, 7, 1.5),
    (28, 8, 1.5),
]


# =============================================================================
# TABLES EMPIRIQUES EXISTANTES
# =============================================================================

BETA_TABLE_MALE = {

    1: [
        (-3, 24.3, 22.7, 18.1),
        (-2, 27.1, 25.1, 19.4),
        (-1, 29.5, 28.1, 20.7),
        (0, 32.0, 31.2, 22.2),
        (1, 35.4, 34.5, 23.8),
        (2, 38.2, 37.9, 25.6),
        (3, 41.1, 41.7, 28.7),
    ],

    2: [
        (-3, 27.5, 26.9, 19.6),
        (-2, 28.9, 28.7, 20.1),
        (-1, 30.2, 29.7, 21.3),
        (0, 32.0, 31.2, 22.2),
        (1, 34.3, 32.9, 24.1),
        (2, 37.1, 33.7, 27.1),
        (3, 39.7, 34.8, 31.2),
    ],
}


BETA_TABLE_FEMALE = {

    1: [
        (-3, 20.5, 19.5, 17.3),
        (-2, 23.3, 23.0, 18.7),
        (-1, 27.6, 26.5, 19.9),
        (0, 32.6, 30.1, 21.9),
        (1, 37.0, 33.8, 24.3),
        (2, 41.4, 37.6, 26.3),
        (3, 46.2, 41.4, 28.9),
    ],

    2: [
        (-3, 37.3, 33.7, 27.6),
        (-2, 35.8, 32.5, 25.3),
        (-1, 34.0, 31.3, 23.1),
        (0, 32.6, 30.1, 21.9),
        (1, 30.5, 29.0, 20.9),
        (2, 28.2, 27.8, 20.2),
        (3, 26.9, 26.7, 19.8),
    ],
}


def _get_beta_table(sex: str):

    return (
        BETA_TABLE_FEMALE
        if sex == "female"
        else BETA_TABLE_MALE
    )


# =============================================================================
# FITTER
# =============================================================================

class SMPLXFitter:

    def __init__(self, model_path: str):

        data = np.load(
            model_path,
            allow_pickle=True,
        )

        self.v_template = np.array(
            data["v_template"]
        )

        self.shapedirs = np.array(
            data["shapedirs"]
        )

        self.faces = np.array(
            data["f"],
            dtype=np.int32,
        )

        jr = data["J_regressor"]

        self.J_regressor = (
            jr.toarray()
            if hasattr(jr, "toarray")
            else np.array(jr)
        )

        self.n_shape = (
            self.shapedirs.shape[2]
        )

    # =========================================================================
    # MESH / JOINTS
    # =========================================================================

    def verts(
        self,
        betas,
        scale=1.0,
    ):

        b = np.zeros(
            self.n_shape
        )

        b[
            :len(betas)
        ] = betas

        return (
            self.v_template
            + np.einsum(
                "ijk,k->ij",
                self.shapedirs,
                b,
            )
        ) * scale

    def joints(
        self,
        betas,
    ):

        return (
            self.J_regressor
            @ self.verts(betas)
        )

    # =========================================================================
    # LARGEUR DU MESH À UNE HAUTEUR NORMALISÉE
    # =========================================================================

    def width_at_normalized_y(
        self,
        vertices,
        y_norm,
        window_ratio=0.012,
    ):
        """
        Largeur X du mesh à une hauteur normalisée.

        y_norm :
            0 = bas
            1 = haut

        On utilise une bande autour du niveau cible pour éviter
        les artefacts liés à une coupe trop fine.
        """

        y_min = vertices[:, 1].min()
        y_max = vertices[:, 1].max()

        height = y_max - y_min

        if height <= 0:
            return 0.0

        y_target = (
            y_min
            + y_norm * height
        )

        window = max(
            height * window_ratio,
            0.001,
        )

        band = vertices[
            np.abs(
                vertices[:, 1]
                - y_target
            ) < window
        ]

        if len(band) < 10:
            return 0.0

        return float(
            band[:, 0].max()
            - band[:, 0].min()
        )

    # =========================================================================
    # PHASE 1 — FACE
    # =========================================================================

    def fit_face(
        self,
        skeleton: dict,
        height_cm: float,
        n_betas: int = 10,
        max_iter: int = 350,
    ):

        scale = _compute_scale(
            skeleton,
            height_cm,
        )

        raw = skeleton.get(
            "raw_proportions",
            {},
        )

        shoulder_target = (
            raw.get(
                "shoulder_width_m",
                0,
            )
            * scale
        )

        hip_target = (
            raw.get(
                "hip_width_m",
                0,
            )
            * scale
        )

        # =====================================================================
        # SILHOUETTE FACE
        # =====================================================================

        front_silhouette = (
            skeleton.get(
                "front_silhouette",
                {},
            )
        )

        shoulder_silhouette_m = (
            float(
                front_silhouette.get(
                    "shoulder_width_m",
                    0,
                )
            )
        )

        shoulder_y_norm = (
            float(
                front_silhouette.get(
                    "shoulder_y_norm",
                    0.83,
                )
            )
        )

        profile = (
            front_silhouette.get(
                "profile",
                [],
            )
        )

        # Convertit le profil silhouette en cibles métriques.
        #
        # On a directement :
        #
        #     largeur_px / hauteur_silhouette_px * taille_m
        #
        # donc aucune hypothèse supplémentaire sur les proportions
        # épaule-cheville.
        silhouette_targets = []

        silhouette_height_px = (
            float(
                front_silhouette.get(
                    "silhouette_height_px",
                    0,
                )
            )
        )

        if (
            silhouette_height_px > 20
            and len(profile) > 0
        ):

            height_m = (
                height_cm / 100.0
            )

            for point in profile:

                width_px = float(
                    point.get(
                        "width_px",
                        0,
                    )
                )

                y_norm = float(
                    point.get(
                        "y_norm",
                        0,
                    )
                )

                if width_px <= 0:
                    continue

                target_width_m = (
                    width_px
                    / silhouette_height_px
                    * height_m
                )

                silhouette_targets.append(
                    (
                        y_norm,
                        target_width_m,
                    )
                )

        print(
            f"  [step2] "
            f"Shoulder silhouette = "
            f"{shoulder_silhouette_m:.3f}m"
        )

        print(
            f"  [step2] "
            f"Shoulder Y = "
            f"{shoulder_y_norm:.3f}"
        )

        print(
            f"  [step2] "
            f"Profil silhouette = "
            f"{len(silhouette_targets)} points"
        )

        # =====================================================================
        # TARGETS MEDIAPIPE
        # =====================================================================

        targets = {}
        weights = {}

        for (
            mp_idx,
            smpl_idx,
            bw,
        ) in MP_TO_SMPLX:

            lm, vis = _get_lm(
                skeleton,
                mp_idx,
            )

            ls = lm * scale

            targets[smpl_idx] = (
                np.array([
                    ls[0],
                    -ls[1],
                    ls[2],
                ])
            )

            weights[smpl_idx] = (
                bw
                * max(vis, 0.3)
            )

        hip_c = (
            targets.get(
                1,
                np.zeros(3),
            )
            + targets.get(
                2,
                np.zeros(3),
            )
        ) / 2.0

        sym_pairs = [
            (16, 17),
            (1, 2),
            (4, 5),
            (7, 8),
        ]

        # =====================================================================
        # LOSS
        # =====================================================================

        def loss(betas):

            jts = self.joints(
                betas
            )

            vts = self.verts(
                betas,
                scale,
            )

            hip_j = (
                jts[1]
                + jts[2]
            ) / 2.0

            jc = (
                jts
                - hip_j
            )

            err = 0.0

            # ---------------------------------------------------------------
            # Joints MediaPipe
            # ---------------------------------------------------------------

            for idx, tgt in targets.items():

                err += (
                    weights[idx]
                    * np.linalg.norm(
                        jc[idx]
                        - (tgt - hip_c)
                    )
                )

            # ---------------------------------------------------------------
            # Symétrie
            # ---------------------------------------------------------------

            for l, r in sym_pairs:

                if (
                    l < len(jc)
                    and r < len(jc)
                ):

                    se = (
                        jc[l]
                        - np.array([
                            -jc[r][0],
                            jc[r][1],
                            jc[r][2],
                        ])
                    )

                    err += (
                        0.5
                        * np.sum(se ** 2)
                    )

            # ---------------------------------------------------------------
            # Largeur globale épaules
            # ---------------------------------------------------------------

            if shoulder_target > 0:

                sw = (
                    np.max(vts[:, 0])
                    - np.min(vts[:, 0])
                )

                err += (
                    1.5
                    * (
                        sw
                        - shoulder_target
                    ) ** 2
                )

            # ---------------------------------------------------------------
            # Largeur hanches
            # ---------------------------------------------------------------

            if hip_target > 0:

                hm = vts[
                    vts[:, 1]
                    < np.percentile(
                        vts[:, 1],
                        60,
                    )
                ]

                if len(hm) > 10:

                    hw = (
                        np.max(hm[:, 0])
                        - np.min(hm[:, 0])
                    )

                    err += (
                        1.0
                        * (
                            hw
                            - hip_target
                        ) ** 2
                    )

            # ---------------------------------------------------------------
            # PROFIL DE SILHOUETTE
            #
            # C'est ici que la vraie largeur du corps observée sur la photo
            # vient compléter MediaPipe.
            # ---------------------------------------------------------------

            if silhouette_targets:

                silhouette_error = 0.0

                for (
                    y_norm,
                    target_width,
                ) in silhouette_targets:

                    model_width = (
                        self.width_at_normalized_y(
                            vts,
                            y_norm,
                        )
                    )

                    if model_width <= 0:
                        continue

                    # Poids maximal autour de l'épaule.
                    distance_from_shoulder = (
                        abs(
                            y_norm
                            - shoulder_y_norm
                        )
                    )

                    shoulder_weight = max(
                        0.4,
                        1.0
                        - distance_from_shoulder
                        / 0.18,
                    )

                    silhouette_error += (
                        shoulder_weight
                        * (
                            model_width
                            - target_width
                        ) ** 2
                    )

                # Poids volontairement modéré :
                #
                # MediaPipe reste utile pour les articulations,
                # mais la silhouette corrige la morphologie réelle.
                err += (
                    4.0
                    * silhouette_error
                )

            # ---------------------------------------------------------------
            # Régularisation des betas
            # ---------------------------------------------------------------

            err += (
                0.3
                * np.sum(
                    betas ** 2
                )
            )

            return float(err)

        # =====================================================================
        # OPTIMISATION
        # =====================================================================

        res = minimize(
            loss,
            np.zeros(n_betas),
            method="L-BFGS-B",
            options={
                "maxiter": max_iter,
            },
            bounds=[
                (-2, 2)
                for _ in range(n_betas)
            ],
        )

        print(
            f"  [step2] "
            f"face loss={res.fun:.3f} | "
            f"betas="
            f"{[round(b, 3) for b in res.x.tolist()]}"
        )

        return (
            res.x,
            scale,
            shoulder_y_norm,
        )

    # =========================================================================
    # PHASE 2 — PROFIL
    # =========================================================================

    def correct_depth(
        self,
        betas,
        scale,
        depth_targets_cm,
        sex="male",
    ):

        target_ventre = depth_targets_cm.get(
            "belly",
            32.0,
        )

        target_hanches = depth_targets_cm.get(
            "hip",
            22.2,
        )

        print(
            f"  [step2] cibles → "
            f"ventre={target_ventre:.1f}cm "
            f"hanches={target_hanches:.1f}cm"
        )

        table = _get_beta_table(
            sex
        )

        b1 = float(
            np.interp(
                target_ventre,
                [
                    r[1]
                    for r in table[1]
                ],
                [
                    r[0]
                    for r in table[1]
                ],
            )
        )

        b_test = betas.copy()

        b_test[1] = b1

        vt = self.verts(
            b_test,
            scale,
        )

        torso = vt[
            np.abs(vt[:, 0]) < 0.15
        ]

        y_min = np.percentile(
            torso[:, 1],
            2,
        )

        yr = (
            np.percentile(
                torso[:, 1],
                98,
            )
            - y_min
        )

        band_h = torso[
            (
                torso[:, 1]
                > y_min + 0.28 * yr
            )
            &
            (
                torso[:, 1]
                < y_min + 0.45 * yr
            )
        ]

        hip_actual = (
            (
                band_h[:, 2].max()
                - band_h[:, 2].min()
            )
            * 100
            if len(band_h) > 10
            else 22.2
        )

        sign = (
            -1.0
            if sex == "female"
            else 1.0
        )

        b2 = float(
            np.clip(
                betas[2]
                + sign
                * (
                    target_hanches
                    - hip_actual
                )
                * 0.15,
                -3.0,
                3.0,
            )
        )

        corrected = betas.copy()

        corrected[1] = b1
        corrected[2] = b2

        print(
            f"  [step2] "
            f"beta[1]={b1:.3f} "
            f"beta[2]={b2:.3f}"
        )

        return corrected


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def fit_smpl(
    skeleton_path: str,
    height_cm: float,
    sex: str,
    skeleton_side_path: str = None,
) -> dict:
    """
    Fit le modèle SMPL-X depuis une ou deux photos.
    """

    model_path = SMPL_MODELS[sex]

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Modèle SMPL-X manquant : "
            f"{model_path}"
        )

    with open(
        skeleton_path,
        encoding="utf-8",
    ) as f:

        skeleton = json.load(f)

    fitter = SMPLXFitter(
        model_path
    )

    # =========================================================================
    # PHASE 1
    # =========================================================================

    print(
        "  [step2] "
        "Phase 1 — fitting face + silhouette..."
    )

    (
        betas,
        scale,
        shoulder_y_norm,
    ) = fitter.fit_face(
        skeleton,
        height_cm,
    )

    # =========================================================================
    # PHASE 2
    # =========================================================================

    if (
        skeleton_side_path
        and os.path.exists(
            skeleton_side_path
        )
    ):

        print(
            "  [step2] "
            "Phase 2 — correction profondeur (profil)..."
        )

        with open(
            skeleton_side_path,
            encoding="utf-8",
        ) as f:

            skeleton_side = json.load(f)

        dp = skeleton_side.get(
            "depth_proportions",
            {},
        )

        src = dp.get(
            "source",
            "",
        )

        if src == "silhouette_side":

            lm_s = skeleton_side.get(
                "landmarks",
                {},
            )

            def ly(i):

                return (
                    lm_s
                    .get(
                        str(i),
                        {},
                    )
                    .get(
                        "pixel_y",
                        0,
                    )
                )

            body_px = max(
                abs(
                    ly(28)
                    - ly(11)
                ),
                1,
            )

            px_per_m = (
                body_px
                / (
                    (height_cm / 100.0)
                    * 0.85
                )
            )

            img_w = skeleton_side.get(
                "image_width",
                426,
            )

            def n2cm(norm):

                return (
                    norm
                    * img_w
                    / px_per_m
                    * 100
                )

            depth_targets_cm = {

                "belly":
                    min(
                        40.0,
                        n2cm(
                            dp.get(
                                "belly_depth_norm",
                                0,
                            )
                        ),
                    ),

                "chest":
                    min(
                        37.0,
                        n2cm(
                            dp.get(
                                "chest_depth_norm",
                                0,
                            )
                        ),
                    ),

                "hip":
                    min(
                        28.0,
                        n2cm(
                            dp.get(
                                "hip_depth_norm",
                                0,
                            )
                        )
                        * 0.65,
                    ),
            }

            betas = fitter.correct_depth(
                betas,
                scale,
                depth_targets_cm,
                sex=sex,
            )

        else:

            print(
                f"  [step2] "
                f"Source={src}, "
                f"pas de correction profondeur"
            )

    else:

        print(
            "  [step2] "
            "Pas de photo profil, "
            "forme face uniquement"
        )

    # =========================================================================
    # INFOS SILHOUETTE POUR LES METRIQUES
    # =========================================================================

    front_silhouette = skeleton.get(
        "front_silhouette",
        {},
    )

    shoulder_width_cm = (
        float(
            front_silhouette.get(
                "shoulder_width_m",
                0,
            )
        )
        * 100
    )

    return {

        "betas":
            betas.tolist(),

        "scale_factor":
            float(scale),

        "height_cm":
            height_cm,

        "sex":
            sex,

        "model_path":
            model_path,

        "shoulder_y_norm":
            float(
                shoulder_y_norm
            ),

        "shoulder_width_silhouette_cm":
            float(
                shoulder_width_cm
            ),
    }


# =============================================================================
# HELPERS
# =============================================================================

def _get_lm(
    skeleton: dict,
    idx: int,
):

    lm = skeleton[
        "landmarks"
    ][str(idx)]

    vis = float(
        lm.get(
            "visibility",
            0.3,
        )
    )

    return (
        np.array([
            lm["world_x"],
            lm["world_y"],
            lm["world_z"],
        ]),
        vis,
    )


def _compute_scale(
    skeleton: dict,
    height_cm: float,
):

    nose, _ = _get_lm(
        skeleton,
        0,
    )

    ankl, _ = _get_lm(
        skeleton,
        27,
    )

    denominator = (
        abs(
            nose[1]
            - ankl[1]
        )
        * 1.1
    )

    if denominator <= 1e-8:
        return 1.0

    return (
        height_cm / 100.0
    ) / denominator
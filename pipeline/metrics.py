"""
pipeline/metrics.py — Calcul des métriques corporelles
=======================================================
Prend un mesh SMPL-X (vertices + faces) + données utilisateur
et retourne un dict de métriques (body fat, BMR, mensurations…)

Mensurations historiques :
    coupes transversales réelles via trimesh.intersections

Nouvelles mesures :
    - circonférence épaules = ellipse face + profil
    - tour de ventre       = ellipse face + profil au niveau du nombril

IMPORTANT :
    waist_cm reste la mesure historique et continue d'être utilisée
    pour les métriques métaboliques.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, asdict


# =============================================================================
# CONFIGURATION
# =============================================================================

# Niveau approximatif du nombril entre les épaules et les hanches.
#
# 0.00 = niveau des épaules
# 1.00 = niveau des hanches
#
# 0.88 = environ niveau du nombril sur les photos du protocole.
BELLY_LEVEL_RATIO = 0.88


# =============================================================================
# STRUCTURES
# =============================================================================

@dataclass
class BodyMeasurements:

    waist_cm: float
    hip_cm: float
    chest_cm: float
    neck_cm: float
    thigh_cm: float

    shoulder_width_cm: float
    shoulder_circumference_cm: float

    # Nouvelle mesure :
    # circonférence au niveau du nombril
    belly_cm: float

    height_cm: float
    bmi: float


@dataclass
class BodyMetrics:

    body_fat_percent: float
    fat_mass_kg: float
    lean_mass_kg: float
    muscle_mass_kg: float
    bone_mass_kg: float
    bmr_kcal: float
    visceral_fat_index: float
    whr: float

    measurements: BodyMeasurements

    confidence_score: float


# =============================================================================
# RATIOS ANTHROPOMETRIQUES DE SECOURS
# =============================================================================

ANTHROPOMETRIC_RATIOS = {

    "male": {

        "waist_to_height":
            0.455,

        "hip_to_height":
            0.540,

        "chest_to_height":
            0.530,

        "neck_to_height":
            0.215,

        "thigh_to_height":
            0.305,

        "shoulder_to_height":
            0.259,

        "shoulder_circumference_to_height":
            0.520,
    },

    "female": {

        "waist_to_height":
            0.430,

        "hip_to_height":
            0.560,

        "chest_to_height":
            0.510,

        "neck_to_height":
            0.205,

        "thigh_to_height":
            0.320,

        "shoulder_to_height":
            0.238,

        "shoulder_circumference_to_height":
            0.500,
    },
}


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def compute_metrics(

    vertices: np.ndarray,

    faces: np.ndarray,

    height_cm: float,

    weight_kg: float,

    age: int,

    sex: str,

    joints: np.ndarray = None,

    skeleton_face: dict = None,

    skeleton_side: dict = None,
    activity_level=0,

) -> dict:

    """
    Calcule toutes les métriques corporelles.

    Les mensurations historiques restent calculées depuis
    le mesh SMPL-X.

    Les deux nouvelles mesures utilisent les silhouettes :

        face + profil → ellipse
    """

    measurements = _extract_measurements(

        vertices,
        faces,
        height_cm,
        weight_kg,
        sex,
        joints,

    )

    # =========================================================================
    # NOUVELLES MESURES PAR SILHOUETTE
    # =========================================================================

    silhouette_measurements = (
        _extract_silhouette_measurements(

            skeleton_face,
            skeleton_side,

            height_cm,

        )
    )

    # -------------------------------------------------------------------------
    # CIRCONFERENCE EPAULES
    # -------------------------------------------------------------------------

    if (
        silhouette_measurements[
            "shoulder_circumference_cm"
        ]
        > 0
    ):

        measurements.shoulder_circumference_cm = (
            silhouette_measurements[
                "shoulder_circumference_cm"
            ]
        )

    # -------------------------------------------------------------------------
    # TOUR DE VENTRE AU NOMBRIL
    # -------------------------------------------------------------------------

    if (
        silhouette_measurements[
            "belly_cm"
        ]
        > 0
    ):

        measurements.belly_cm = (
            silhouette_measurements[
                "belly_cm"
            ]
        )

    # =========================================================================
    # METRIQUES METABOLIQUES
    # =========================================================================

    # IMPORTANT :
    #
    # On utilise TOUJOURS waist_cm historique ici.
    #
    # belly_cm ne remplace PAS waist_cm.
    #
    metrics = _compute_metabolic(
    measurements,
    weight_kg,
    age,
    sex,
    activity_level=activity_level)

    return asdict(
        metrics
    )


# =============================================================================
# EXTRACTION MENSURATIONS HISTORIQUES
# =============================================================================

def _extract_measurements(

    vertices: np.ndarray,

    faces: np.ndarray,

    height_cm: float,

    weight_kg: float,

    sex: str,

    joints: np.ndarray = None,

) -> BodyMeasurements:

    import trimesh

    # =========================================================================
    # RECALIBRAGE DU MESH A LA TAILLE REELLE
    # =========================================================================

    mesh_height = (
        vertices[:, 1].max()
        -
        vertices[:, 1].min()
    )

    if mesh_height > 0:

        scale = (
            (height_cm / 100.0)
            /
            mesh_height
        )

        verts = (
            vertices
            *
            scale
        )

    else:

        verts = (
            vertices.copy()
        )

        scale = 1.0

    mesh = trimesh.Trimesh(

        vertices=verts,

        faces=faces,

        process=False,

    )

    y_min = (
        verts[:, 1].min()
    )

    y_max = (
        verts[:, 1].max()
    )

    h_m = (
        y_max
        -
        y_min
    )

    # =========================================================================
    # NIVEAU TAILLE
    # =========================================================================
    #
    # CONSERVE TON CALCUL ORIGINAL :
    #
    # milieu crête iliaque / dernière côte
    #

    if (
        joints is not None
        and
        len(joints) > 6
    ):

        j = (
            joints
            *
            scale
        )

        y_pelvis = (
            j[0, 1]
        )

        y_spine2 = (
            j[6, 1]
        )

        y_waist = (
            y_pelvis
            +
            y_spine2
        ) / 2.0

    else:

        y_waist = (
            y_min
            +
            h_m * 0.615
        )

    # =========================================================================
    # AUTRES NIVEAUX ANATOMIQUES
    # =========================================================================

    y_hip = (
        y_min
        +
        h_m * 0.520
    )

    y_chest = (
        y_min
        +
        h_m * 0.720
    )

    y_neck = (
        y_min
        +
        h_m * 0.895
    )

    y_thigh = (
        y_min
        +
        h_m * 0.420
    )

    # =========================================================================
    # CIRCONFERENCES HISTORIQUES
    # =========================================================================

    waist_c = _circumference(
        mesh,
        y_waist
    )

    hip_c = _circumference(
        mesh,
        y_hip
    )

    chest_c = _circumference(
        mesh,
        y_chest
    )

    neck_c = _circumference(
        mesh,
        y_neck
    )

    thigh_c = _circumference_half(
        mesh,
        y_thigh
    )

    # =========================================================================
    # EPAULES — LARGEUR HISTORIQUE DU MESH
    # =========================================================================

    y_sho = (
        y_min
        +
        h_m * 0.830
    )

    band = verts[
        np.abs(
            verts[:, 1]
            -
            y_sho
        )
        < 0.025
    ]

    shoulder_w = (

        float(
            band[:, 0].max()
            -
            band[:, 0].min()
        )
        *
        100.0

        if len(band) > 5

        else 0.0
    )

    # =========================================================================
    # EPAULES — FALLBACK HISTORIQUE
    # =========================================================================

    shoulder_circ = _circumference(
        mesh,
        y_sho
    )

    # =========================================================================
    # FALLBACKS ANTHROPOMETRIQUES
    # =========================================================================

    bmi = (
        weight_kg
        /
        (height_cm / 100.0) ** 2
    )

    ratios = (
        ANTHROPOMETRIC_RATIOS[
            sex
        ]
    )

    def safe(

        val,

        ratio_key,

        lo=20.0,

        hi=200.0,

    ):

        if (
            val
            and
            lo < val < hi
        ):

            return round(
                val,
                1
            )

        fallback = round(

            height_cm
            *
            ratios[
                ratio_key
            ],

            1,

        )

        print(

            f"  [metrics] ⚠ "
            f"{ratio_key} hors limites "
            f"({val:.1f}cm) → "
            f"fallback "
            f"{fallback:.1f}cm"

        )

        return fallback

    # =========================================================================
    # RESULTAT
    # =========================================================================

    result = BodyMeasurements(

        waist_cm=
            safe(
                waist_c,
                "waist_to_height",
                40,
                150,
            ),

        hip_cm=
            safe(
                hip_c,
                "hip_to_height",
                50,
                170,
            ),

        chest_cm=
            safe(
                chest_c,
                "chest_to_height",
                50,
                170,
            ),

        neck_cm=
            safe(
                neck_c,
                "neck_to_height",
                20,
                60,
            ),

        thigh_cm=
            safe(
                thigh_c,
                "thigh_to_height",
                30,
                100,
            ),

        shoulder_width_cm=
            safe(
                shoulder_w,
                "shoulder_to_height",
                25,
                70,
            ),

        shoulder_circumference_cm=
            safe(
                shoulder_circ,
                "shoulder_circumference_to_height",
                60,
                200,
            ),

        # Sera remplacé juste après par
        # la mesure ellipse si les deux masques existent.
        belly_cm=
            0.0,

        height_cm=
            height_cm,

        bmi=
            round(
                bmi,
                2
            ),
    )

    print(
        f"  [metrics] taille mesh    : "
        f"{h_m * 100:.1f} cm"
    )

    print(
        f"  [metrics] tour taille    : "
        f"{result.waist_cm:.1f} cm"
    )

    print(
        f"  [metrics] tour hanches   : "
        f"{result.hip_cm:.1f} cm"
    )

    print(
        f"  [metrics] tour poitrine  : "
        f"{result.chest_cm:.1f} cm"
    )

    print(
        f"  [metrics] tour cou       : "
        f"{result.neck_cm:.1f} cm"
    )

    print(
        f"  [metrics] tour cuisse    : "
        f"{result.thigh_cm:.1f} cm"
    )

    print(
        f"  [metrics] largeur épaules: "
        f"{result.shoulder_width_cm:.1f} cm"
    )

    print(
        f"  [metrics] tour d'épaule  : "
        f"{result.shoulder_circumference_cm:.1f} cm"
    )

    return result


# =============================================================================
# NOUVELLES MESURES SILHOUETTE
# =============================================================================

def _extract_silhouette_measurements(

    skeleton_face,

    skeleton_side,

    height_cm,

):

    result = {

        "shoulder_circumference_cm":
            0.0,

        "belly_cm":
            0.0,

    }

    if (
        not skeleton_face
        or
        not skeleton_side
    ):

        return result

    # =========================================================================
    # MASQUES
    # =========================================================================

    face_mask = _load_mask(
        skeleton_face
    )

    side_mask = _load_mask(
        skeleton_side
    )

    if (
        face_mask is None
        or
        side_mask is None
    ):

        print(
            "  [metrics] ⚠ "
            "Masque face/profil indisponible"
        )

        return result

    # =========================================================================
    # HAUTEURS DE SILHOUETTE
    # =========================================================================

    face_height_px = float(

        skeleton_face.get(
            "silhouette_height_px",
            0
        )
    )

    side_height_px = float(

        skeleton_side.get(
            "silhouette_height_px",
            0
        )
    )

    if (
        face_height_px <= 0
        or
        side_height_px <= 0
        or
        height_cm <= 0
    ):

        return result

    # =========================================================================
# EPAULES — légèrement sous le point MediaPipe
# =========================================================================

    SHOULDER_OFFSET_RATIO = 0.025
    
    face_landmarks = (
        skeleton_face.get(
            "landmarks",
            {}
        )
    )
    
    y_left_shoulder = _landmark_y(
        face_landmarks,
        11
    )
    
    y_right_shoulder = _landmark_y(
        face_landmarks,
        12
    )
    
    shoulder_width_px = 0.0
    
    if (
        y_left_shoulder > 0
        and
        y_right_shoulder > 0
    ):
    
        # Y moyen des deux épaules MediaPipe
        y_shoulder = (
            y_left_shoulder
            +
            y_right_shoulder
        ) / 2.0
    
        # On descend légèrement sous le landmark MediaPipe
        y_shoulder = (
            y_shoulder
            +
            SHOULDER_OFFSET_RATIO
            *
            face_height_px
        )
    
        # On mesure directement la largeur du masque
        # à ce nouveau niveau
        shoulder_width_px = (
            _width_at_y_px(
                face_mask,
                y_shoulder,
                window=5
            )
        )

    # =========================================================================
    # PROFONDEUR EPAULES (pareil, un peu en dessous du point mediapipe)
    # =========================================================================

    
    
    side_landmarks = (
        skeleton_side.get(
            "landmarks",
            {}
        )
    )
    
    y_left_shoulder_side = _landmark_y(
        side_landmarks,
        11
    )
    
    y_right_shoulder_side = _landmark_y(
        side_landmarks,
        12
    )
    
    shoulder_depth_px = 0.0
    
    if (
        y_left_shoulder_side > 0
        or
        y_right_shoulder_side > 0
    ):
    
        valid = [
            y
            for y in [
                y_left_shoulder_side,
                y_right_shoulder_side,
            ]
            if y > 0
        ]
    
        # Y moyen des deux épaules MediaPipe
        y_shoulder_side = (
            sum(valid)
            /
            len(valid)
        )
    
        # On descend légèrement sous le landmark MediaPipe
        y_shoulder_side = (
            y_shoulder_side
            +
            SHOULDER_OFFSET_RATIO
            *
            side_height_px
        )
    
        shoulder_depth_px = (
            _width_at_y_px(
                side_mask,
                y_shoulder_side
            )
        )
    # =========================================================================
    # CONVERSION EPAULES → CM
    # =========================================================================

    if (
        shoulder_width_px > 0
        and
        shoulder_depth_px > 0
    ):

        shoulder_width_cm = (

            shoulder_width_px
            /
            face_height_px
            *
            height_cm
        )

        shoulder_depth_cm = (

            shoulder_depth_px
            /
            side_height_px
            *
            height_cm
        )

        shoulder_circ = (
            _ellipse_circumference(

                shoulder_width_cm,

                shoulder_depth_cm,

            )
        )

        result[
            "shoulder_circumference_cm"
        ] = round(
            shoulder_circ,
            1
        )

        print(
            f"  [metrics] "
            f"épaule largeur face : "
            f"{shoulder_width_cm:.1f} cm"
        )

        print(
            f"  [metrics] "
            f"épaule profondeur profil : "
            f"{shoulder_depth_cm:.1f} cm"
        )

        print(
            f"  [metrics] "
            f"épaule ellipse : "
            f"{shoulder_circ:.1f} cm"
        )

    # =========================================================================
    # VENTRE — NIVEAU NOMBRIL
    # =========================================================================
    #
    # On ne prend PAS belly_depth_norm.
    #
    # On calcule directement le même niveau anatomique
    # dans les deux images :
    #
    #       épaules
    #           ↓
    #        88 %
    #           ↓
    #         hanches
    #
    # Cela garantit que largeur et profondeur correspondent
    # au même niveau du corps.
    # =========================================================================

    belly_face_width_px = _mask_width_at_belly_level(

        face_mask,

        skeleton_face,

        BELLY_LEVEL_RATIO,

    )

    belly_side_depth_px = _mask_width_at_belly_level(

        side_mask,

        skeleton_side,

        BELLY_LEVEL_RATIO,

    )

    if (
        belly_face_width_px > 0
        and
        belly_side_depth_px > 0
    ):

        belly_width_cm = (

            belly_face_width_px
            /
            face_height_px
            *
            height_cm
        )

        belly_depth_cm = (

            belly_side_depth_px
            /
            side_height_px
            *
            height_cm
        )

        belly_circ = (
            _ellipse_circumference(

                belly_width_cm,

                belly_depth_cm,

            )
        )

        result[
            "belly_cm"
        ] = round(
            belly_circ,
            1
        )

        print(
            f"  [metrics] "
            f"ventre largeur face : "
            f"{belly_width_cm:.1f} cm"
        )

        print(
            f"  [metrics] "
            f"ventre profondeur profil : "
            f"{belly_depth_cm:.1f} cm"
        )

        print(
            f"  [metrics] "
            f"tour ventre ellipse : "
            f"{belly_circ:.1f} cm"
        )

    else:

        print(
            "  [metrics] ⚠ "
            "Impossible de calculer "
            "le tour de ventre ellipse"
        )

    return result


# =============================================================================
# CHARGEMENT MASQUE
# =============================================================================

def _load_mask(
    skeleton
):

    mask_path = (
        skeleton.get(
            "mask_path",
            ""
        )
    )

    if (
        not mask_path
        or
        not os.path.exists(
            mask_path
        )
    ):

        return None

    mask = cv2.imread(

        mask_path,

        cv2.IMREAD_GRAYSCALE

    )

    if mask is None:

        return None

    return (
        mask > 127
    ).astype(
        np.uint8
    )


# =============================================================================
# LANDMARK Y
# =============================================================================

def _landmark_y(
    landmarks,
    idx
):

    lm = landmarks.get(
        str(idx),
        {}
    )

    return float(
        lm.get(
            "pixel_y",
            0
        )
    )


# =============================================================================
# LARGEUR MASQUE A UN Y
# =============================================================================

def _width_at_y_px(

    binary_mask,

    y_px,

    window=8

):

    if (
        binary_mask is None
        or
        binary_mask.size == 0
    ):

        return 0.0

    h = (
        binary_mask.shape[0]
    )

    y = max(

        window,

        min(
            h - 1 - window,
            int(round(y_px))
        )

    )

    widths = []

    for dy in range(

        -window,

        window + 1

    ):

        row = (
            binary_mask[
                y + dy,
                :
            ]
        )

        xs = np.where(
            row > 0
        )[0]

        if len(xs) > 5:

            widths.append(

                float(
                    xs.max()
                    -
                    xs.min()
                )
            )

    if not widths:

        return 0.0

    return float(
        np.median(
            widths
        )
    )


# =============================================================================
# LARGEUR DU MASQUE AU NIVEAU DU VENTRE
# =============================================================================
def _mask_width_at_belly_level(
    binary_mask,
    skeleton,
    ratio,
):
    """
    Mesure la largeur du TORSE au niveau du ventre.

    IMPORTANT :
    En vue de face, les bras sont légèrement écartés du corps.
    À certaines hauteurs, ils apparaissent donc comme des composantes
    séparées du torse.

    On ne prend PAS la largeur totale du masque.

    On prend la composante centrale correspondant au torse.
    """

    landmarks = skeleton.get(
        "landmarks",
        {}
    )

    # -------------------------------------------------------------------------
    # Y EPAULES
    # -------------------------------------------------------------------------

    shoulder_values = [

        _landmark_y(
            landmarks,
            11
        ),

        _landmark_y(
            landmarks,
            12
        ),

    ]

    shoulder_values = [

        y
        for y in shoulder_values
        if y > 0
    ]

    # -------------------------------------------------------------------------
    # Y HANCHES
    # -------------------------------------------------------------------------

    hip_values = [

        _landmark_y(
            landmarks,
            23
        ),

        _landmark_y(
            landmarks,
            24
        ),

    ]

    hip_values = [

        y
        for y in hip_values
        if y > 0
    ]

    if (
        not shoulder_values
        or
        not hip_values
    ):

        return 0.0

    y_shoulder = (
        sum(shoulder_values)
        /
        len(shoulder_values)
    )

    y_hip = (
        sum(hip_values)
        /
        len(hip_values)
    )

    # -------------------------------------------------------------------------
    # Y DU NOMBRIL
    # -------------------------------------------------------------------------

    y_belly = (

        y_shoulder

        +

        ratio
        *
        (
            y_hip
            -
            y_shoulder
        )

    )

    # -------------------------------------------------------------------------
    # FENETRE AUTOUR DU NOMBRIL
    # -------------------------------------------------------------------------

    h = binary_mask.shape[0]

    y_center = max(
        5,
        min(
            h - 6,
            int(round(y_belly))
        )
    )

    widths = []

    # -------------------------------------------------------------------------
    # ANALYSE DE PLUSIEURS LIGNES
    # -------------------------------------------------------------------------

    for dy in range(
        -8,
        9
    ):

        y = y_center + dy

        row = binary_mask[
            y,
            :
        ]

        xs = np.where(
            row > 0
        )[0]

        if len(xs) < 5:
            continue

        # ---------------------------------------------------------------------
        # SEGMENTS CONTIGUS
        # ---------------------------------------------------------------------

        breaks = np.where(
            np.diff(xs) > 1
        )[0]

        starts = np.r_[

            0,

            breaks + 1

        ]

        ends = np.r_[

            breaks,

            len(xs) - 1

        ]

        segments = []

        for start, end in zip(
            starts,
            ends
        ):

            x1 = int(
                xs[start]
            )

            x2 = int(
                xs[end]
            )

            width = (
                x2 - x1
            )

            if width >= 8:

                segments.append(
                    (
                        x1,
                        x2,
                        width
                    )
                )

        if not segments:
            continue

        # ---------------------------------------------------------------------
        # COMPOSANTE CENTRALE
        # ---------------------------------------------------------------------
        #
        # On cherche le segment dont le centre est le plus proche
        # du centre horizontal de l'image.
        #

        image_center = (
            binary_mask.shape[1]
            /
            2.0
        )

        central_segment = min(

            segments,

            key=lambda seg:

                abs(
                    (
                        seg[0]
                        +
                        seg[1]
                    )
                    /
                    2.0
                    -
                    image_center
                )

        )

        widths.append(
            float(
                central_segment[2]
            )
        )

    if not widths:

        return 0.0

    # Médiane pour être robuste aux petites irrégularités
    return float(
        np.median(
            widths
        )
    )


# =============================================================================
# ELLIPSE
# =============================================================================

def _ellipse_circumference(

    width_cm: float,

    depth_cm: float,

) -> float:

    """
    Circonférence approximative d'une ellipse.

    width_cm :
        diamètre / largeur vue de face.

    depth_cm :
        diamètre / profondeur vue de profil.

    Approximation de Ramanujan II.
    """

    if (
        width_cm <= 0
        or
        depth_cm <= 0
    ):

        return 0.0

    a = (
        width_cm
        /
        2.0
    )

    b = (
        depth_cm
        /
        2.0
    )

    h = (

        (a - b) ** 2

        /

        (a + b) ** 2

    )

    circumference = (

        np.pi
        *
        (a + b)
        *
        (

            1.0

            +

            (

                3.0
                *
                h

                /

                (
                    10.0
                    +
                    np.sqrt(
                        4.0
                        -
                        3.0 * h
                    )
                )

            )

        )

    )

    return float(
        circumference
    )


# =============================================================================
# CIRCONFERENCE MESH — ORIGINAL
# =============================================================================

def _circumference(

    mesh,

    y_target: float

) -> float:

    """
    Coupe transversale réelle du mesh au niveau y_target.

    Retourne le périmètre total en cm.
    """

    import trimesh

    plane_origin = np.array(

        [
            0.0,
            y_target,
            0.0
        ]

    )

    plane_normal = np.array(

        [
            0.0,
            1.0,
            0.0
        ]

    )

    try:

        lines = (
            trimesh.intersections.mesh_plane(

                mesh,

                plane_normal,

                plane_origin,

            )
        )

        if (
            lines is None
            or
            len(lines) == 0
        ):

            return 0.0

        lengths = np.linalg.norm(

            lines[:, 1]
            -
            lines[:, 0],

            axis=1,

        )

        return float(
            lengths.sum()
        ) * 100

    except Exception as e:

        print(

            f"  [metrics] "
            f"avertissement coupe "
            f"y={y_target:.3f} : "
            f"{e}"

        )

        return 0.0


# =============================================================================
# CIRCONFERENCE CUISSE — ORIGINAL
# =============================================================================

def _circumference_half(

    mesh,

    y_target: float

) -> float:

    """
    Coupe transversale pour la cuisse :
    garde uniquement le côté droit (x > 0)
    pour éviter de sommer les deux jambes.
    """

    import trimesh

    plane_origin = np.array(

        [
            0.0,
            y_target,
            0.0
        ]

    )

    plane_normal = np.array(

        [
            0.0,
            1.0,
            0.0
        ]

    )

    try:

        lines = (
            trimesh.intersections.mesh_plane(

                mesh,

                plane_normal,

                plane_origin,

            )
        )

        if (
            lines is None
            or
            len(lines) == 0
        ):

            return 0.0

        # Segments du côté droit uniquement

        right = lines[
            lines[:, 0, 0] > 0.01
        ]

        if len(right) == 0:

            lengths = np.linalg.norm(

                lines[:, 1]
                -
                lines[:, 0],

                axis=1,

            )

            return (
                float(
                    lengths.sum()
                )
                *
                100
                /
                2
            )

        lengths = np.linalg.norm(

            right[:, 1]
            -
            right[:, 0],

            axis=1,

        )

        return float(
            lengths.sum()
        ) * 100

    except Exception as e:

        print(

            f"  [metrics] "
            f"avertissement cuisse "
            f"y={y_target:.3f}: "
            f"{e}"

        )

        return 0.0


# =============================================================================
# METRIQUES METABOLIQUES — CONSERVEES
# =============================================================================

def _compute_metabolic(

    m: BodyMeasurements,

    weight_kg: float,

    age: int,

    sex: str,
    
    activity_level: int = 0,

) -> BodyMetrics:
    # =========================================================================
    # NIVEAU D'ACTIVITE
    # =========================================================================
    
    activity_level = int(activity_level)
    
    if activity_level not in (0, 1, 2, 3, 4,5 ):
        raise ValueError(
            "activity_level doit être 0, 1, 2, 3, 4,5"
        )
    
    # Proportion de masse maigre considérée comme musculaire
   # Proportion de masse maigre considérée comme musculaire
    MUSCLE_RATIO = {
        0: 0.50,   # Léger / actif sédentaire — musculation
        1: 0.55,   # Léger / actif sédentaire — endurance
        2: 0.53,   # Léger actif sédentaire - un peu des 2
        3: 0.58,   # Grand sportif — musculation
        4: 0.60,   # Grand sportif — endurance
        5: 0.59    # Grand sportif - un peu des 2
    }
    
    # Correction de l'estimation de masse grasse
    FAT_ADJUSTMENT = {
        0: 0.10,  # Léger/actif sédentaire — musculation
        1: 0.20,  # Léger/ actif sédentaire — endurance
        2: 0.15,   # Léger / actif sédentaire - un peu des 2
        3: 0.33,  # Grand sportif — musculation
        4: 0.40,  # Grand sportif — endurance
        5: 0.36,  # Grand sportif - un peu des 2
       
    }
        
    bf = _body_fat(
    m,
    age,
    sex
)

    # Ajustement lié au niveau d'activité
    bf = bf * (
        1.0
        -
        FAT_ADJUSTMENT[activity_level]
    )

    fat_mass = (
        weight_kg
        *
        bf
        /
        100
    )

    lean_mass = (
        weight_kg
        -
        fat_mass
    )

    muscle_ratio = MUSCLE_RATIO[
    activity_level
]

    # Pour les femmes, on conserve une proportion
    # légèrement inférieure à celle des hommes.
    if sex == "female":
        muscle_ratio -= 0.02
    
    muscle = (
        lean_mass
        *
        muscle_ratio
    )

    bone = _bone_mass(
        weight_kg
    )

    bmr = _bmr_black(

        weight_kg,

        m.height_cm,

        age,

        sex,

    )

    vfi = _visceral_fat(
        m,
        sex
    )

    whr = (
        m.waist_cm
        /
        m.hip_cm
    )

    return BodyMetrics(

        body_fat_percent=
            round(
                bf,
                1
            ),

        fat_mass_kg=
            round(
                fat_mass,
                2
            ),

        lean_mass_kg=
            round(
                lean_mass,
                2
            ),

        muscle_mass_kg=
            round(
                muscle,
                2
            ),

        bone_mass_kg=
            round(
                bone,
                2
            ),

        bmr_kcal=
            round(
                bmr
            ),

        visceral_fat_index=
            round(
                vfi,
                1
            ),

        whr=
            round(
                whr,
                3
            ),

        measurements=
            m,

        confidence_score=
            0.65,
    )


# =============================================================================
# MASSE GRASSE — ORIGINAL
# =============================================================================

def _body_fat(

    m: BodyMeasurements,

    age: int,

    sex: str

) -> float:

    """
    Deurenberg et al. (1991) + ajustement WHR.
    """

    sex_c = (
        1
        if sex == "male"
        else 0
    )

    bf = (

        1.20
        *
        m.bmi

        +

        0.23
        *
        age

        -

        10.8
        *
        sex_c

        -

        5.4

    )

    whr = (

        m.waist_cm
        /
        m.hip_cm

    )

    bf += max(

        -5.0,

        min(

            5.0,

            (
                whr
                -
                (
                    0.90
                    if sex == "male"
                    else 0.80
                )
            )
            *
            15

        )

    )

    return max(

        3.0,

        min(
            60.0,
            bf
        )

    )


# =============================================================================
# MASSE OSSEUSE — ORIGINAL
# =============================================================================

def _bone_mass(

    w: float

) -> float:

    if w < 45:

        return 1.8

    if w < 75:

        return 2.5

    if w < 95:

        return 3.2

    return 3.8


# =============================================================================
# BMR — ORIGINAL
# =============================================================================

def _bmr_mifflin(

    w: float,

    h: float,

    age: int,

    sex: str

) -> float:

    """
    Mifflin-St Jeor (1990).
    """

    bmr = (

        10 * w

        +

        6.25 * h

        -

        5 * age

    )

    return (

        bmr + 5

        if sex == "male"

        else

        bmr - 161

    )

def _bmr_black(w: float, h_cm: float, age: int, sex: str) -> float:
    """
    Black, Coward, Cole, Prentice (1996) — basée sur méta-analyse eau doublement
    marquée. Remplace Mifflin-St Jeor : exposant sur l'âge (^-0.13) plutôt qu'un
    terme linéaire, jugée plus précise en population générale.
 
    NB UNITÉS : la formule originale attend T en MÈTRES avec une constante de
    conversion (1000/4.1855). Ici T = h_cm est utilisé directement en CM (cohérent
    avec le reste du fichier), donc la constante a été divisée par 10 (1000 -> 100)
    pour compenser : T_cm^0.5 = (T_m × 100)^0.5 = T_m^0.5 × 10.
    Équivalent mathématiquement à la formule source, vérifié numériquement
    (70kg/175cm/30ans homme -> ~1690 kcal dans les deux versions).
    """
    age   = max(age, 1)   # évite age**-0.13 indéfini/instable si age<=0
    coeff = 1.083 if sex == "male" else 0.963
    mb_mj = coeff * (w ** 0.48) * (h_cm ** 0.50) * (age ** -0.13)
    return mb_mj * (100.0 / 4.1855)



# =============================================================================
# GRAISSE VISCERALE — ORIGINAL
# =============================================================================

def _visceral_fat(

    m: BodyMeasurements,

    sex: str

) -> float:

    baseline = (

        36.76
        if sex == "male"
        else 36.18
    )

    vfi = (

        m.waist_cm
        /
        baseline

    ) * (

        m.waist_cm
        /
        (
            m.hip_cm
            *
            0.6
        )

    ) - 1.6

    return max(

        1.0,

        min(
            59.0,
            vfi * 5
        )

    )
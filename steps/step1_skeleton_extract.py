"""
STEP 1 — Extraction du squelette 3D via MediaPipe
==================================================
Entrée  : chemin image (face ou profil)
Sortie  : skeleton.json (33 landmarks x/y/z + visibilité + métriques brutes)
"""

import json
import os
import cv2
import numpy as np
import mediapipe as mp


LANDMARK_NAMES = {
    0:  "nose",
    1:  "left_eye_inner",
    2:  "left_eye",
    3:  "left_eye_outer",
    4:  "right_eye_inner",
    5:  "right_eye",
    6:  "right_eye_outer",
    7:  "left_ear",
    8:  "right_ear",
    9:  "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),

    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),

    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),

    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),

    (9, 10),
    (11, 0),
    (12, 0),
]


def _width_at_y_px(
    binary_mask: np.ndarray,
    y_px: int,
    window: int = 8
) -> float:

    h = binary_mask.shape[0]

    y = max(
        window,
        min(
            h - 1 - window,
            int(y_px)
        )
    )

    widths = []

    for dy in range(
        -window,
        window + 1
    ):

        row = binary_mask[
            y + dy,
            :
        ]

        xs = np.where(
            row > 0
        )[0]

        if len(xs) > 5:

            widths.append(
                float(
                    xs.max()
                    - xs.min()
                )
            )

    return (
        float(np.median(widths))
        if widths
        else 0.0
    )


def _extract_depth_metrics(
    landmarks_3d,
    landmarks_2d,
    binary_mask,
    w,
    h
):

    metrics = {}

    l_sho = landmarks_3d[11]
    r_sho = landmarks_3d[12]

    is_side_view = (
        abs(
            l_sho.x
            - r_sho.x
        ) < 0.15
    )

    metrics["view_type"] = (
        "side"
        if is_side_view
        else "front"
    )

    if (
        is_side_view
        and binary_mask is not None
        and binary_mask.sum() > 100
    ):

        def y_px(idx):
            return int(
                landmarks_2d[idx].y
                * h
            )

        # ---------------------------------------------------------------------
        # Y DES EPAULES
        # ---------------------------------------------------------------------

        # Profil : MediaPipe n'a pas une vraie séparation gauche/droite
        # exploitable comme en vue de face.
        #
        # On utilise donc le Y de l'épaule visible.
        y_shoulder = y_px(11)

        y_hip = y_px(23)

        # ---------------------------------------------------------------------
        # LARGEUR DU MASQUE A UN Y
        # ---------------------------------------------------------------------

        def width_at_y(
            y,
            window=8
        ):

            y = max(
                window,
                min(
                    binary_mask.shape[0]
                    - 1
                    - window,
                    y
                )
            )

            widths = []

            for dy in range(
                -window,
                window + 1
            ):

                row = binary_mask[
                    y + dy,
                    :
                ]

                xs = np.where(
                    row > 0
                )[0]

                if len(xs) > 5:

                    widths.append(
                        float(
                            xs.max()
                            - xs.min()
                        )
                        / binary_mask.shape[1]
                    )

            return (
                float(
                    np.median(widths)
                )
                if widths
                else 0.0
            )

        # ---------------------------------------------------------------------
        # SCAN DU TORSE
        # ---------------------------------------------------------------------

        torse_range = max(
            y_hip - y_shoulder,
            1
        )

        scan = {
            pct:
                width_at_y(
                    int(
                        y_shoulder
                        + torse_range
                        * pct
                        / 100
                    )
                )
            for pct in range(
                0,
                101,
                3
            )
        }

        mid_vals = [
            scan[p]
            for p in range(
                35,
                66,
                3
            )
            if p in scan
        ]

        low_vals = [
            scan[p]
            for p in range(
                15,
                36,
                3
            )
            if p in scan
        ]

        high_vals = [
            scan[p]
            for p in range(
                65,
                86,
                3
            )
            if p in scan
        ]

        has_waist_dip = (
            bool(
                mid_vals
                and low_vals
                and high_vals
            )
            and
            min(mid_vals)
            < max(low_vals)
            * 0.92
            and
            min(mid_vals)
            < max(high_vals)
            * 0.92
        )

        if has_waist_dip:

            chest_pct = max(
                range(
                    15,
                    42,
                    3
                ),
                key=lambda p:
                    scan.get(p, 0)
            )

            waist_pct = min(
                range(
                    36,
                    66,
                    3
                ),
                key=lambda p:
                    scan.get(p, 1)
            )

            hip_pct = max(
                range(
                    60,
                    87,
                    3
                ),
                key=lambda p:
                    scan.get(p, 0)
            )

            belly_pct = hip_pct

        else:

            belly_pct = max(
                range(
                    39,
                    75,
                    3
                ),
                key=lambda p:
                    scan.get(p, 0)
            )

            chest_pct = max(
                range(
                    15,
                    42,
                    3
                ),
                key=lambda p:
                    scan.get(p, 0)
            )

            waist_pct = min(
                range(
                    36,
                    66,
                    3
                ),
                key=lambda p:
                    scan.get(p, 1)
            )

            hip_pct = max(
                range(
                    60,
                    87,
                    3
                ),
                key=lambda p:
                    scan.get(p, 0)
            )

        metrics.update({

            "chest_depth_norm":
                scan.get(
                    chest_pct,
                    0
                ),

            "waist_depth_norm":
                scan.get(
                    waist_pct,
                    0
                ),

            "hip_depth_norm":
                scan.get(
                    hip_pct,
                    0
                ),

            "belly_depth_norm":
                scan.get(
                    belly_pct,
                    0
                ),

            # IMPORTANT :
            # profondeur au niveau des épaules
            "shoulder_depth_norm":
                width_at_y(
                    y_shoulder
                ),

            "has_waist_dip":
                has_waist_dip,

            "source":
                "silhouette_side",
        })

    else:

        avg_z = (
            lambda a, b:
            abs(
                landmarks_3d[a].z
                + landmarks_3d[b].z
            ) / 2
        )

        metrics.update({

            "chest_depth_norm":
                avg_z(11, 12),

            "waist_depth_norm":
                avg_z(23, 24),

            "hip_depth_norm":
                avg_z(23, 24)
                * 1.1,

            "belly_depth_norm":
                avg_z(23, 24)
                * 1.05,

            "shoulder_depth_norm":
                avg_z(11, 12),

            "source":
                "mediapipe_z_estimated",
        })

    return metrics


def extract_skeleton(
    image_path: str,
    output_path: str,
    debug_image: bool = True
) -> str:

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image introuvable : "
            f"{image_path}"
        )

    img_bgr = cv2.imread(
        image_path
    )

    if img_bgr is None:

        raise ValueError(
            f"Impossible de lire "
            f"l'image : {image_path}"
        )

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    h, w = img_bgr.shape[:2]

    print(
        f"  [step1] Image chargée : "
        f"{w}x{h}px"
    )

    # =========================================================================
    # SEGMENTATION
    # =========================================================================

    mp_selfie = (
        mp.solutions.selfie_segmentation
    )

    with mp_selfie.SelfieSegmentation(
        model_selection=1
    ) as segmenter:

        results_seg = (
            segmenter.process(
                img_rgb
            )
        )

        binary_mask = (
            results_seg.segmentation_mask
            > 0.5
        ).astype(
            np.uint8
        )

    # =========================================================================
    # POSE
    # =========================================================================

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.4,
    ) as pose:

        print(
            "  [step1] Détection de pose..."
        )

        results = pose.process(
            img_rgb
        )

    if not results.pose_landmarks:

        raise RuntimeError(
            "Aucune personne détectée. "
            "Conseils : fond uni, corps entier visible."
        )

    landmarks_2d = (
        results.pose_landmarks.landmark
    )

    landmarks_3d = (
        results.pose_world_landmarks.landmark
    )

    # =========================================================================
    # FACE OU PROFIL
    # =========================================================================

    is_front_view = (
        abs(
            landmarks_3d[11].x
            - landmarks_3d[12].x
        )
        >= 0.15
    )

    view_type = (
        "front"
        if is_front_view
        else "side"
    )

    print(
        f"  [step1] Vue détectée : "
        f"{view_type}"
    )

    # =========================================================================
    # DIMENSIONS SILHOUETTE
    # =========================================================================

    ys, xs = np.where(
        binary_mask > 0
    )

    width_px = (
        int(
            xs.max()
            - xs.min()
        )
        if len(xs) > 0
        else 0
    )

    height_px = (
        int(
            ys.max()
            - ys.min()
        )
        if len(ys) > 0
        else 0
    )

    # =========================================================================
    # SAUVEGARDE MASQUE
    # =========================================================================

    debug_dir = os.path.dirname(
        output_path
    )

    os.makedirs(
        debug_dir,
        exist_ok=True
    )

    if is_front_view:

        mask_filename = (
            "mask_face.png"
        )

    else:

        mask_filename = (
            "mask_profil.png"
        )

    mask_path = os.path.join(
        debug_dir,
        mask_filename
    )

    cv2.imwrite(
        mask_path,
        binary_mask * 255
    )

    print(
        f"  [step1] Masque → "
        f"{mask_path}"
    )

    # =========================================================================
    # SKELETON
    # =========================================================================

    skeleton = {

        "image_path":
            image_path,

        "image_width":
            w,

        "image_height":
            h,

        "view_type":
            view_type,

        "mask_path":
            mask_path,

        "silhouette_width_px":
            width_px,

        "silhouette_height_px":
            height_px,

        "landmarks":
            {},

        "connections":
            SKELETON_CONNECTIONS,
    }

    # =========================================================================
    # LANDMARKS
    # =========================================================================

    for i, (lm2d, lm3d) in enumerate(
        zip(
            landmarks_2d,
            landmarks_3d
        )
    ):

        name = LANDMARK_NAMES.get(
            i,
            f"point_{i}"
        )

        skeleton[
            "landmarks"
        ][str(i)] = {

            "name":
                name,

            "pixel_x":
                round(
                    lm2d.x * w,
                    2
                ),

            "pixel_y":
                round(
                    lm2d.y * h,
                    2
                ),

            "world_x":
                round(
                    lm3d.x,
                    6
                ),

            "world_y":
                round(
                    lm3d.y,
                    6
                ),

            "world_z":
                round(
                    lm3d.z,
                    6
                ),

            "visibility":
                round(
                    lm2d.visibility,
                    3
                ),
        }

    # =========================================================================
    # PROPORTIONS BRUTES
    # =========================================================================

    def lm(i):

        return np.array([
            landmarks_3d[i].x,
            landmarks_3d[i].y,
            landmarks_3d[i].z
        ])

    def dist(a, b):

        return float(
            np.linalg.norm(
                lm(a) - lm(b)
            )
        )

    skeleton[
        "raw_proportions"
    ] = {

        "shoulder_width_m":
            dist(11, 12),

        "hip_width_m":
            dist(23, 24),

        "torso_height_m":
            dist(11, 23),

        "left_arm_m":
            dist(11, 13)
            + dist(13, 15),

        "right_arm_m":
            dist(12, 14)
            + dist(14, 16),

        "left_leg_m":
            dist(23, 25)
            + dist(25, 27),

        "right_leg_m":
            dist(24, 26)
            + dist(26, 28),

        "left_femur_m":
            dist(23, 25),

        "left_tibia_m":
            dist(25, 27),
    }

    # =========================================================================
    # LARGEUR EPAULES PAR SEGMENTATION FACE
    # =========================================================================

    if (
        is_front_view
        and binary_mask.sum() > 100
    ):

        # Y moyen des deux épaules MediaPipe
        y_shoulder_px = (
            (
                landmarks_2d[11].y
                +
                landmarks_2d[12].y
            )
            / 2.0
            * h
        )

        shoulder_w_px = (
            _width_at_y_px(
                binary_mask,
                y_shoulder_px
            )
        )

        # Calibration historique conservée
        calib_px = (
            abs(
                landmarks_2d[11].y
                -
                landmarks_2d[28].y
            )
            * h
        )

        if (
            shoulder_w_px > 0
            and calib_px > 5
        ):

            skeleton[
                "raw_proportions"
            ][
                "shoulder_width_silhouette_px"
            ] = shoulder_w_px

            skeleton[
                "raw_proportions"
            ][
                "shoulder_calib_px"
            ] = calib_px

            skeleton[
                "shoulder_face"
            ] = {

                "y_px":
                    float(
                        y_shoulder_px
                    ),

                "width_px":
                    float(
                        shoulder_w_px
                    ),

                "calib_px":
                    float(
                        calib_px
                    ),
            }

            print(
                f"  [step1] "
                f"Largeur épaules "
                f"silhouette = "
                f"{shoulder_w_px:.1f}px"
            )

    # =========================================================================
    # PROFONDEURS
    # =========================================================================

    skeleton[
        "depth_proportions"
    ] = _extract_depth_metrics(
        landmarks_3d,
        landmarks_2d,
        binary_mask,
        w,
        h
    )

    # =========================================================================
    # SAUVEGARDE
    # =========================================================================

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            skeleton,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"  [step1] "
        f"Squelette exporté → "
        f"{output_path}"
    )

    # =========================================================================
    # DEBUG
    # =========================================================================

    if debug_image:

        debug_path = (
            output_path.replace(
                ".json",
                "_debug.jpg"
            )
        )

        _draw_skeleton_debug(
            img_bgr,
            results,
            debug_path
        )

    return output_path


def _draw_skeleton_debug(
    img_bgr,
    results,
    output_path: str
):

    mp_drawing = (
        mp.solutions.drawing_utils
    )

    mp_pose = (
        mp.solutions.pose
    )

    img_copy = img_bgr.copy()

    mp_drawing.draw_landmarks(

        img_copy,

        results.pose_landmarks,

        mp_pose.POSE_CONNECTIONS,

        mp_drawing.DrawingSpec(
            color=(124, 108, 255),
            thickness=2,
            circle_radius=4
        ),

        mp_drawing.DrawingSpec(
            color=(255, 108, 157),
            thickness=2
        ),
    )

    cv2.imwrite(
        output_path,
        img_copy
    )

    print(
        f"  [step1] Debug image → "
        f"{output_path}"
    )
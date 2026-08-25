"""
STEP 3 — Export Avatar 3D (GLB)
================================
Entrée  : dict params SMPL-X (betas, scale, sex, model_path) + pose optionnelle
Sortie  : fichier .glb prêt pour affichage mobile
"""

import os
import json
import numpy as np
import trimesh
from PIL import Image


# Couleur peau par défaut 
SKIN_COLOR_RGB = (77, 171, 139)   # #4DAB8B personnalisable


# Poses disponibles 
def _t_pose():
    import torch
    return torch.zeros((1, 63))


def _relaxed():
    import torch
    pose = torch.zeros((1, 63))
    pose[0, 15*3 + 2] = -1.2
    pose[0, 16*3 + 2] =  1.2
    pose[0, 15*3 + 0] = -0.2
    pose[0, 16*3 + 0] = -0.2
    pose[0, 17*3 + 0] = -0.3
    pose[0, 18*3 + 0] = -0.3
    pose[0, 12*3 + 2] =  0.1
    pose[0, 13*3 + 2] = -0.1
    pose[0,  2*3 + 0] =  0.05
    return pose


POSES = {
    "tpose":   _t_pose,
    "relaxed": _relaxed,
}


# Export principal
def export_avatar(params: dict, output_path: str, pose: str = "tpose") -> str:
    """
    Génère un fichier GLB depuis les paramètres SMPL-X.

    Args:
        params:      dict issu de step2 (betas, scale_factor, sex, model_path)
        output_path: chemin du .glb à créer
        pose:        nom de la pose ('tpose' | 'relaxed')

    Returns:
        output_path
    """
    import torch
    import smplx

    betas      = np.array(params["betas"])
    scale      = float(params["scale_factor"])
    sex        = params["sex"]
    model_path = params["model_path"]

    # Chargement mod�le
    model = smplx.create(
        model_path,
        model_type="smplx",
        gender=sex,
        use_pca=False,
        batch_size=1,
    )

    betas_t = torch.tensor(betas).float().unsqueeze(0)

    # Pose 
    pose_fn = POSES.get(pose, _t_pose)
    body_pose     = pose_fn()
    global_orient = torch.zeros((1, 3))
    left_hand     = torch.zeros((1, 45))
    right_hand    = torch.zeros((1, 45))

    print(f"  [step3] Pose : {pose}")

    # g�n�ration du mesh
    print("  [step3] Generation du mesh...")
    with torch.no_grad():
        output = model(
            betas=betas_t,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            return_verts=True,
        )

    vertices = output.vertices.detach().cpu().numpy()[0] * scale
    faces    = model.faces

    # Export GLB 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _export_glb(vertices, faces, model_path, output_path)

    print(f"  [step3] GLB exporté → {output_path}")
    return output_path


#  Helpers

def _create_skin_texture(tex_size: int = 1024) -> np.ndarray:
    texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    texture[:] = SKIN_COLOR_RGB
    noise   = np.random.normal(0, 3, texture.shape).astype(np.int32)
    return np.clip(texture + noise, 0, 255).astype(np.uint8)


def _export_glb(vertices: np.ndarray, faces: np.ndarray,
                model_path: str, output_path: str):
    """Construit le mesh avec UV + texture et l'exporte en GLB."""
    data = np.load(model_path, allow_pickle=True)
    vt   = data["vt"]
    ft   = data["ft"]

    # D�duplication UV (un vertex par triangle)
    new_verts = vertices[faces.reshape(-1)]
    new_faces = np.arange(len(faces) * 3).reshape(len(faces), 3)
    new_uvs   = vt[ft.reshape(-1)]
    new_uvs[:, 1] = 1.0 - new_uvs[:, 1]   # flip V

    texture = _create_skin_texture()
    mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=new_uvs,
        image=Image.fromarray(texture),
    )
    mesh.export(output_path)

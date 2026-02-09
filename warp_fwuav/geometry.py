"""Geometry helpers for internal wing generation (QS_model.pdf frames)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WingMesh:
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int indices


@dataclass(frozen=True)
class RectangularWing:
    """Rectangular wing in the right-wing frame FR.

    FR axes (QS_model.pdf):
      rx: toward leading edge
      ry: toward wing tip (spanwise)
      rz: ventral (normal to wing plane)

    The wing root is at the origin. The leading edge is located at +root_to_le
    along rx, and the trailing edge is at x = root_to_le - chord.
    """

    chord: float
    span: float
    thickness: float = 0.0
    root_to_le: float = 0.0

    def chord_limits(self) -> tuple[float, float]:
        x_le = self.root_to_le
        x_te = x_le - self.chord
        return x_te, x_le

    def surface_mesh(self, n_chord: int, n_span: int, xp=np) -> WingMesh:
        """Mid-plane surface mesh (no thickness)."""
        x_te, x_le = self.chord_limits()
        xs = xp.linspace(x_te, x_le, n_chord)
        ys = xp.linspace(0.0, self.span, n_span)
        xx, yy = xp.meshgrid(xs, ys)
        zz = xp.zeros_like(xx)
        vertices = xp.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        faces = _grid_faces(n_span, n_chord, offset=0, flip=False)
        return WingMesh(vertices=vertices, faces=faces)

    def solid_mesh(self, n_chord: int, n_span: int, xp=np) -> WingMesh:
        """Watertight thin solid mesh using top/bottom surfaces and side walls."""
        if self.thickness <= 0.0:
            return self.surface_mesh(n_chord, n_span, xp=xp)

        x_te, x_le = self.chord_limits()
        xs = xp.linspace(x_te, x_le, n_chord)
        ys = xp.linspace(0.0, self.span, n_span)
        xx, yy = xp.meshgrid(xs, ys)

        z_top = -0.5 * self.thickness
        z_bot = 0.5 * self.thickness

        top = xp.stack([xx, yy, xp.full_like(xx, z_top)], axis=-1).reshape(-1, 3)
        bot = xp.stack([xx, yy, xp.full_like(xx, z_bot)], axis=-1).reshape(-1, 3)
        vertices = xp.vstack([top, bot])

        n_top = top.shape[0]
        faces = []

        # Top surface (dorsal)
        faces.extend(_grid_faces(n_span, n_chord, offset=0, flip=False))
        # Bottom surface (ventral)
        faces.extend(_grid_faces(n_span, n_chord, offset=n_top, flip=True))

        # Side walls
        def idx_top(i: int, j: int) -> int:
            return i * n_chord + j

        def idx_bot(i: int, j: int) -> int:
            return n_top + i * n_chord + j

        # Leading edge (j = n_chord - 1)
        j = n_chord - 1
        for i in range(n_span - 1):
            v0 = idx_top(i, j)
            v1 = idx_top(i + 1, j)
            v2 = idx_bot(i, j)
            v3 = idx_bot(i + 1, j)
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

        # Trailing edge (j = 0)
        j = 0
        for i in range(n_span - 1):
            v0 = idx_top(i, j)
            v1 = idx_top(i + 1, j)
            v2 = idx_bot(i, j)
            v3 = idx_bot(i + 1, j)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

        # Root edge (i = 0)
        i = 0
        for j in range(n_chord - 1):
            v0 = idx_top(i, j)
            v1 = idx_top(i, j + 1)
            v2 = idx_bot(i, j)
            v3 = idx_bot(i, j + 1)
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

        # Tip edge (i = n_span - 1)
        i = n_span - 1
        for j in range(n_chord - 1):
            v0 = idx_top(i, j)
            v1 = idx_top(i, j + 1)
            v2 = idx_bot(i, j)
            v3 = idx_bot(i, j + 1)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

        return WingMesh(vertices=vertices, faces=np.array(faces, dtype=int))


def _grid_faces(n_span: int, n_chord: int, offset: int, flip: bool) -> np.ndarray:
    """Create two triangles per quad for a structured grid."""
    faces = []
    for i in range(n_span - 1):
        for j in range(n_chord - 1):
            v0 = offset + i * n_chord + j
            v1 = offset + i * n_chord + (j + 1)
            v2 = offset + (i + 1) * n_chord + j
            v3 = offset + (i + 1) * n_chord + (j + 1)
            if flip:
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
            else:
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])
    return np.array(faces, dtype=int)


def transform_vertices(vertices, R, t, xp=np):
    """Apply a rigid transform to vertices."""
    vertices = xp.asarray(vertices)
    R = xp.asarray(R).reshape(3, 3)
    t = xp.asarray(t).reshape(3)
    return (R @ vertices.T).T + t

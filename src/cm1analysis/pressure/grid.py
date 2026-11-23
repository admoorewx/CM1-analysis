"""Grid geometry, z-face interpolation and index boxes.

Arrays are ordered (z, y, x). Cell-centered fields have nz levels and z-face
fields nz + 1. Horizontal spacing is uniform; the vertical grid may be
stretched. Heights come from the file and are never hard-coded.
"""

from dataclasses import dataclass

import numpy as np

EDGE_BUFFER_CELLS = 20
ROI_PAD_KM = 25.0
ROI_TRIM_PERCENT = 1.0


def column(a, ndim=3):
    """Reshape a 1-D profile to (n, 1, 1) so it broadcasts; leave 3-D arrays alone."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(a.shape + (1,) * (ndim - 1))
    return a


class Grid:
    """Uniform horizontal spacing and a stretched vertical grid.

    zf holds the nz + 1 face heights and zc the nz cell-center heights (m).
    dz[k] is the thickness of cell k and dzf[k] the distance between the
    centers on either side of interior face k (NaN on the boundary faces).
    """

    def __init__(self, dx, dy, zf, zc=None):
        self.dx = float(dx)
        self.dy = float(dy)
        self.zf = np.asarray(zf, dtype=np.float64)
        if self.zf.ndim != 1 or self.zf.size < 4:
            raise ValueError("need at least three vertical cells")
        self.dz = np.diff(self.zf)
        if np.any(self.dz <= 0.0):
            raise ValueError("z faces must increase upward")
        if zc is None:
            zc = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.zc = np.asarray(zc, dtype=np.float64)
        if self.zc.shape != self.dz.shape:
            raise ValueError(f"{self.zc.size} centers do not match {self.zf.size} faces")
        if np.any(self.zc <= self.zf[:-1]) or np.any(self.zc >= self.zf[1:]):
            raise ValueError("cell centers must lie strictly inside their cells")
        self.nz = self.zc.size
        self.dzf = np.full(self.nz + 1, np.nan)
        self.dzf[1:-1] = np.diff(self.zc)
        # Weight of the cell below when interpolating linearly to an interior face.
        self._lower = np.full(self.nz + 1, np.nan)
        self._lower[1:-1] = (self.zc[1:] - self.zf[1:-1]) / self.dzf[1:-1]

    def _profile_shape(self, f):
        return (-1,) + (1,) * (f.ndim - 1)

    def to_faces(self, f, boundary="nearest"):
        """Interpolate centers to z-faces, linearly in z.

        Boundary faces copy the adjacent cell ("nearest") or extrapolate the
        lowest or highest pair of centers ("linear").
        """
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.nz:
            raise ValueError(f"expected {self.nz} levels, got {f.shape[0]}")
        w = self._lower[1:-1].reshape(self._profile_shape(f))
        out = np.empty((self.nz + 1,) + f.shape[1:])
        out[1:-1] = w * f[:-1]
        out[1:-1] += (1.0 - w) * f[1:]
        if boundary == "nearest":
            out[0], out[-1] = f[0], f[-1]
        elif boundary == "linear":
            lo = (self.zf[0] - self.zc[0]) / (self.zc[1] - self.zc[0])
            hi = (self.zf[-1] - self.zc[-1]) / (self.zc[-1] - self.zc[-2])
            out[0] = f[0] + (f[1] - f[0]) * lo
            out[-1] = f[-1] + (f[-1] - f[-2]) * hi
        else:
            raise ValueError(f"unknown boundary rule {boundary!r}")
        return out

    def to_faces_log(self, f):
        """Interpolate a positive profile, such as density, linearly in log."""
        return np.exp(self.to_faces(np.log(f), "linear"))

    def faces_to_centers(self, f):
        """Interpolate z-face values (such as CM1 w) linearly to cell centers."""
        f = np.asarray(f, dtype=np.float64)
        if f.shape[0] != self.nz + 1:
            raise ValueError(f"expected {self.nz + 1} faces, got {f.shape[0]}")
        shape = self._profile_shape(f)
        upper = ((self.zc - self.zf[:-1]) / self.dz).reshape(shape)
        out = (1.0 - upper) * f[:-1]
        out += upper * f[1:]
        return out

    def ddz_faces(self, f):
        """(f[k] - f[k-1]) / dzf[k] at interior faces; zero on the boundary faces."""
        f = np.asarray(f, dtype=np.float64)
        out = np.zeros((self.nz + 1,) + f.shape[1:])
        out[1:-1] = (f[1:] - f[:-1]) / self.dzf[1:-1].reshape(self._profile_shape(f))
        return out

    def ddz(self, f):
        """Second-order vertical derivative at centers on the stretched grid."""
        return np.gradient(f, self.zc, axis=0, edge_order=2)

    def ddx(self, f):
        return np.gradient(f, self.dx, axis=-1, edge_order=2)

    def ddy(self, f):
        return np.gradient(f, self.dy, axis=-2, edge_order=2)

    def face_index_below(self, z_top):
        """Highest face index k with zf[k] - zf[0] <= z_top (m)."""
        return int(np.searchsorted(self.zf - self.zf[0], z_top + 1e-6, side="right") - 1)


# ── Index boxes ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Box:
    """Half-open cell index ranges: i0 <= i < i1 and j0 <= j < j1."""

    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def nx(self):
        return self.i1 - self.i0

    @property
    def ny(self):
        return self.j1 - self.j0

    @property
    def cells(self):
        return (slice(self.j0, self.j1), slice(self.i0, self.i1))

    def relative_to(self, outer):
        return Box(self.i0 - outer.i0, self.i1 - outer.i0, self.j0 - outer.j0, self.j1 - outer.j0)

    def contains(self, inner):
        return self.i0 <= inner.i0 and inner.i1 <= self.i1 and self.j0 <= inner.j0 and inner.j1 <= self.j1

    def margins(self, inner):
        """Cells between inner and this box on the west, east, south and north."""
        return (inner.i0 - self.i0, self.i1 - inner.i1, inner.j0 - self.j0, self.j1 - inner.j1)


def roi_from_reflectivity(cref, threshold, dx, dy):
    """Box around cref >= threshold, or None if the threshold is never reached.

    Same rule as the wind decomposition: the 1st to 99th percentile of the
    qualifying cell positions, padded by ROI_PAD_KM and kept EDGE_BUFFER_CELLS
    away from the domain edge.
    """
    ny, nx = cref.shape
    jj, ii = np.nonzero(np.asarray(cref) >= threshold)
    if ii.size == 0:
        return None
    pad_i = int(round(ROI_PAD_KM * 1000.0 / dx))
    pad_j = int(round(ROI_PAD_KM * 1000.0 / dy))
    low, high = ROI_TRIM_PERCENT, 100.0 - ROI_TRIM_PERCENT
    i0 = max(int(np.floor(np.percentile(ii, low))) - pad_i, EDGE_BUFFER_CELLS)
    i1 = min(int(np.ceil(np.percentile(ii, high))) + pad_i + 1, nx - EDGE_BUFFER_CELLS)
    j0 = max(int(np.floor(np.percentile(jj, low))) - pad_j, EDGE_BUFFER_CELLS)
    j1 = min(int(np.ceil(np.percentile(jj, high))) + pad_j + 1, ny - EDGE_BUFFER_CELLS)
    if i1 <= i0 or j1 <= j0:
        return None
    return Box(i0, i1, j0, j1)


def buffered(roi, buffer_i, buffer_j, nx, ny):
    """Grow roi by the buffer on every side, clipped to the domain."""
    return Box(
        max(roi.i0 - buffer_i, 0),
        min(roi.i1 + buffer_i, nx),
        max(roi.j0 - buffer_j, 0),
        min(roi.j1 + buffer_j, ny),
    )


def trimmed(crop, roi, fraction):
    """Shrink the buffer between crop and roi by fraction on every side."""
    west, east, south, north = crop.margins(roi)
    keep = 1.0 - fraction
    return Box(
        roi.i0 - int(round(west * keep)),
        roi.i1 + int(round(east * keep)),
        roi.j0 - int(round(south * keep)),
        roi.j1 + int(round(north * keep)),
    )

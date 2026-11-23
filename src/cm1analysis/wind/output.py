"""Partition output: one netCDF file holding every part of the wind, plus diagnostics.

The file keeps the source's coordinate names, dimensions and global attributes,
stores every field in float64 on its native C-grid location (``psi`` and
``zeta`` on corners, ``chi`` and ``delta`` at cell centers, ``u_*`` on x faces,
``v_*`` on y faces), and holds the validation and sensitivity numbers in a
``diagnostics`` group. It is written to ``<name>.tmp`` and renamed into place
only after a clean close, so a crashed or concurrent job never leaves a
half-written file under the final name.
"""

import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, default_fillvals

from .diagnostics import CROP_FRACTIONS

COMPRESSION = {"zlib": True, "complevel": 1, "shuffle": True}
FIELDS = ("psi", "chi", "u_psi", "v_psi", "u_chi", "v_chi", "u_h", "v_h", "delta")
VARIANT_FIELDS = ("u_rot", "v_rot", "u_div", "v_div")

FIELD_ATTRS = {
    "psi": ("m2 s-1", "streamfunction"),
    "chi": ("m2 s-1", "velocity potential"),
    "u_psi": ("m s-1", "rotational u"),
    "v_psi": ("m s-1", "rotational v"),
    "u_chi": ("m s-1", "divergent u"),
    "v_chi": ("m s-1", "divergent v"),
    "u_h": ("m s-1", "harmonic u"),
    "v_h": ("m s-1", "harmonic v"),
    "delta": ("s-1", "horizontal divergence"),
    "zeta": ("s-1", "vertical vorticity"),
    "u_rot": ("m s-1", "rotational u for the selected variant"),
    "v_rot": ("m s-1", "rotational v for the selected variant"),
    "u_div": ("m s-1", "divergent u for the selected variant"),
    "v_div": ("m s-1", "divergent v for the selected variant"),
}

DIAGNOSTIC_ATTRS = {
    "vpsi_normal_edge_max": ("m s-1", "max |normal V_psi| on edge faces"),
    "chi_ghost_rule_max": ("m s-1", "max error in the chi ghost-cell rule"),
    "vh_net_flux": ("m2 s-1", "net V_h boundary flux per unit depth"),
    "vh_gross_flux": ("m2 s-1", "gross V_h boundary flux per unit depth"),
    "ke_total": ("m2 s-2", "kinetic energy summed over faces"),
    "vh_roi_mean_u": ("m s-1", "ROI-mean harmonic u"),
    "vh_roi_mean_v": ("m s-1", "ROI-mean harmonic v"),
}


class PartitionWriter:
    """Write one output file, renaming it into place only after a clean close.

    ``source`` is a :class:`cm1analysis.wind.source.WindLevel`.
    """

    def __init__(self, path, source, n_times, variant, roi_source, attrs=None):
        self.path = Path(path)
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")
        self.variant = variant
        self.ds = Dataset(self.tmp_path, "w", format="NETCDF4")
        self.diag = self.ds.createGroup("diagnostics")
        self._define_grid(source, n_times)
        self._define_fields(source)
        self._define_attributes(source, roi_source, attrs or {})
        self._diagnostics_defined = False

    def _define_grid(self, source, n_times):
        self.time_dim = source.time_var.dimensions[0]
        self.ds.createDimension(self.time_dim, n_times)
        for coord in (source.x_center, source.x_face, source.y_center, source.y_face):
            self.ds.createDimension(coord.dimensions[0], coord.size)
            var = self.ds.createVariable(coord.name, "f8", coord.dimensions)
            var[:] = np.asarray(coord[:], dtype=np.float64)
            var.units = coord.units
        self.time_out = self.ds.createVariable(source.time_var.name, "f8", (self.time_dim,))
        self.time_out.units = getattr(source.time_var, "units", "seconds")
        self.diag.createDimension("crop", len(CROP_FRACTIONS))
        fractions = self.diag.createVariable("crop_fraction", "f8", ("crop",))
        fractions[:] = CROP_FRACTIONS
        fractions.long_name = "fraction trimmed from every edge"

    def _define_fields(self, source):
        center = (source.y_center_dim, source.x_center_dim)
        x_faces = (source.y_center_dim, source.x_face_dim)
        y_faces = (source.y_face_dim, source.x_center_dim)
        corners = (source.y_face_dim, source.x_face_dim)
        layout = {
            "psi": corners,
            "chi": center,
            "u_psi": x_faces,
            "v_psi": y_faces,
            "u_chi": x_faces,
            "v_chi": y_faces,
            "u_h": x_faces,
            "v_h": y_faces,
            "delta": center,
            "zeta": corners,
        }
        if self.variant != "A":
            layout.update(u_rot=x_faces, v_rot=y_faces, u_div=x_faces, v_div=y_faces)
        for name, dims in layout.items():
            size = [source.ds.dimensions[d].size for d in dims]
            var = self.ds.createVariable(
                name,
                "f8",
                (self.time_dim, *dims),
                chunksizes=(1, *size),
                fill_value=default_fillvals["f8"] if name == "zeta" else None,
                **COMPRESSION,
            )
            var.units, var.long_name = FIELD_ATTRS[name]

    def _define_attributes(self, source, roi_source, attrs):
        for key in source.ds.ncattrs():
            self.ds.setncattr(key, source.ds.getncattr(key))
        self.ds.setncatts(
            {
                "partition_method": "Lynch (1989): psi = chi = 0 on the boundary, residual harmonic",
                "variant": self.variant,
                "source_file": str(source.path),
                "level_index": source.level,
                "level_height_m": source.level_height_m,
                "dx_m": source.dx,
                "dy_m": source.dy,
                "roi_source": roi_source,
                **attrs,
            }
        )

    def _define_diagnostics(self, values):
        for name, value in values.items():
            array = np.asarray(value)
            kind = "i4" if np.issubdtype(array.dtype, np.integer) else "f8"
            dims = (self.time_dim, "crop") if array.ndim == 1 else (self.time_dim,)
            var = self.diag.createVariable(name, kind, dims)
            units, long_name = DIAGNOSTIC_ATTRS.get(name, ("1", name.replace("_", " ")))
            var.units, var.long_name = units, long_name
        self._diagnostics_defined = True

    def write(self, index, time, p, diagnostics, variant_winds=None):
        self.time_out[index] = time
        for name in FIELDS:
            self.ds[name][index] = getattr(p, name)
        zeta = np.ma.masked_all(p.psi.shape, dtype=np.float64)
        zeta[1:-1, 1:-1] = p.zeta
        self.ds["zeta"][index] = zeta
        if self.variant != "A":
            for name, field in zip(VARIANT_FIELDS, variant_winds, strict=True):
                self.ds[name][index] = field
        if not self._diagnostics_defined:
            self._define_diagnostics(diagnostics)
        for name, value in diagnostics.items():
            self.diag[name][index] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ds.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.path)
        elif self.tmp_path.exists():
            self.tmp_path.unlink()

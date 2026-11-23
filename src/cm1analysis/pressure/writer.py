"""Output files of the decomposition (``wforce_*.nc``).

Each file holds the fields of one variant on the region of interest (ROI), the
base-state profiles, composite reflectivity when the source has it, and a
``diagnostics`` group of validation and sensitivity numbers per time. The file
is written under a temporary name and renamed only after a clean close, so a
crash never leaves a partial file that looks complete.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from . import vforce

COMPRESSION = {"zlib": True, "complevel": 1, "shuffle": True}
FIELD_ATTRS = {
    # name: (location, units, long_name)
    "p_d": ("center", "Pa", "dynamic pressure perturbation p'_d"),
    "p_b": ("center", "Pa", "buoyancy pressure perturbation p'_b"),
    "dvpgf": ("face", "m s-2", "dynamic vertical pressure-gradient force -(1/rho0) dp'_d/dz"),
    "bpgf": ("face", "m s-2", "buoyancy pressure-gradient force -(1/rho0) dp'_b/dz"),
    "b": ("face", "m s-2", "buoyancy b at z-faces"),
    "B": ("face", "m s-2", "effective buoyancy b + bpgf"),
    "model_vpgf": ("face", "m s-2", "-(1/rho0) dp'/dz from the CM1 pressure perturbation"),
    "model_vpgf_exner": ("face", "m s-2", "CM1 vertical PGF -c_p theta_v0 dpi'/dz from the model pressure"),
    "w": ("face", "m s-1", "vertical velocity"),
    "bpgf_thermal": ("face", "m s-2", "buoyancy PGF of the thermal (theta', qv') part"),
    "bpgf_loading": ("face", "m s-2", "buoyancy PGF of the loading (-g qw) part"),
    "B_thermal": ("face", "m s-2", "effective buoyancy, thermal part"),
    "B_loading": ("face", "m s-2", "effective buoyancy, loading part"),
    "dvpgf_shear": ("face", "m s-2", "DVPGF from the shear source (RK85), level mean removed"),
    "dvpgf_extension": ("face", "m s-2", "DVPGF from the extension source (RK85), level mean removed"),
    "dvpgf_shear_remainder": ("face", "m s-2", "DVPGF remainder of the shear/extension split"),
    "dvpgf_rotation": ("face", "m s-2", "DVPGF from rho |omega|^2 / 2 (T&W), level mean removed"),
    "dvpgf_strain": ("face", "m s-2", "DVPGF from -rho e_ij e_ij (T&W), level mean removed"),
    "dvpgf_density": ("face", "m s-2", "DVPGF from rho (ln rho)'' w^2, level mean removed"),
    "dvpgf_strain_remainder": ("face", "m s-2", "DVPGF remainder of the strain/rotation split"),
    # Horizontal forces, on the ROI's x-faces and y-faces at cell-center levels.
    "hdpgf_x": ("xface", "m s-2", "horizontal dynamic PGF -(1/rho0) dp'_d/dx"),
    "hdpgf_y": ("yface", "m s-2", "horizontal dynamic PGF -(1/rho0) dp'_d/dy"),
    "hbpgf_x": ("xface", "m s-2", "horizontal buoyancy PGF -(1/rho0) dp'_b/dx"),
    "hbpgf_y": ("yface", "m s-2", "horizontal buoyancy PGF -(1/rho0) dp'_b/dy"),
    "model_hpgf_x": ("xface", "m s-2", "-(1/rho0) dp'/dx from the CM1 pressure perturbation"),
    "model_hpgf_y": ("yface", "m s-2", "-(1/rho0) dp'/dy from the CM1 pressure perturbation"),
    "hdpgf_s": ("center", "m s-2", "HDPGF along the horizontal wind, (u F_x + v F_y) / |V_h|"),
    "hbpgf_s": ("center", "m s-2", "HBPGF along the horizontal wind, (u F_x + v F_y) / |V_h|"),
}

DIAGNOSTIC_ATTRS = {
    "poisson_residual_dynamic": ("1", "max |L p_d - rhs| / max |rhs|"),
    "poisson_residual_buoyancy": ("1", "max |L p_b - rhs| / max |rhs|"),
    "solvability_dynamic": ("1", "|integrated rhs| / integrated |rhs|, dynamic"),
    "solvability_buoyancy": ("1", "|integrated rhs| / integrated |rhs|, buoyancy"),
    "hydrostatic_residual_p": ("1", "max |dp0/dz + g rho0 (1 + qv0)| / max g rho0 (1 + qv0)"),
    "hydrostatic_residual_exner": ("1", "max |dpi0/dz + g / (c_p theta_v0)| / max g / (c_p theta_v0)"),
    "base_th0_spread": ("K", "max horizontal spread of th - thpert"),
    "base_qv0_spread": ("kg kg-1", "max horizontal spread of qv - qvpert"),
    "base_p0_spread": ("Pa", "max horizontal spread of prs - prspert"),
    "consistency_force_rel_rms": ("1", "RMS(DVPGF + B - model force) / RMS(model force) in the ROI"),
    "consistency_force_corr": ("1", "correlation of DVPGF + B with -(1/rho0) dp'/dz + b"),
    "consistency_exner_rel_rms": (
        "1",
        "RMS(DVPGF + B - CM1 force) / RMS(CM1 force), CM1 force -c_p theta_v0 dpi'/dz + b",
    ),
    "consistency_exner_corr": ("1", "correlation of DVPGF + B with -c_p theta_v0 dpi'/dz + b"),
    "consistency_pressure_rel_rms": ("1", "RMS(p_d + p_b - p') / RMS(p'), level means removed"),
    "anelastic_residual": ("1", "RMS div(rho0 V) / RMS of its terms in the ROI"),
    "frame_u": ("m s-1", "frame-shift u for the reference-frame test"),
    "frame_v": ("m s-1", "frame-shift v for the reference-frame test"),
    "frame_dvpgf_rel": ("1", "relative RMS change of DVPGF in the shifted frame"),
    "roi_rms_dvpgf": ("m s-2", "RMS DVPGF in the ROI"),
    "roi_rms_B": ("m s-2", "RMS B in the ROI"),
    "crop_buffer_cells": ("1", "smallest remaining buffer after trimming"),
    "crop_dvpgf_rel": ("1", "relative RMS change of DVPGF in the ROI after trimming"),
    "crop_B_rel": ("1", "relative RMS change of B in the ROI after trimming"),
    "frame_hdpgf_rel": ("1", "relative RMS change of HDPGF (x and y) in the shifted frame"),
    "roi_rms_hdpgf": ("m s-2", "RMS HDPGF (x and y faces) in the ROI"),
    "roi_rms_hbpgf": ("m s-2", "RMS HBPGF (x and y faces) in the ROI"),
    "roi_rms_model_hpgf": ("m s-2", "RMS -(1/rho0) grad_h p' (x and y faces) in the ROI"),
    "crop_hdpgf_rel": ("1", "relative RMS change of HDPGF in the ROI after trimming"),
    "crop_hbpgf_rel": ("1", "relative RMS change of HBPGF in the ROI after trimming"),
    "crop_hcons_rel_rms": ("1", "RMS(HDPGF + HBPGF - model force) / RMS(model force) after trimming"),
}
for _v in vforce.VARIANTS:
    DIAGNOSTIC_ATTRS.update(
        {
            f"hcons_{_v}_rel_rms": ("1", f"variant {_v}: RMS(HDPGF + HBPGF - model) / RMS(model), x and y"),
            f"hcons_{_v}_corr": ("1", f"variant {_v}: correlation of HDPGF + HBPGF with -(1/rho0) grad_h p'"),
            f"hcons_{_v}_bias_x": ("m s-2", f"variant {_v}: mean(HDPGF_x + HBPGF_x - model x force)"),
            f"hcons_{_v}_bias_y": ("m s-2", f"variant {_v}: mean(HDPGF_y + HBPGF_y - model y force)"),
            f"sens_hdpgf_{_v}": ("1", f"relative RMS difference of HDPGF, variant {_v} vs A"),
            f"sens_hbpgf_{_v}": ("1", f"relative RMS difference of HBPGF, variant {_v} vs A"),
        }
    )


class WForceWriter:
    """Write one output file on the ROI, renaming it into place only after a clean close."""

    def __init__(self, path, source, roi, crop, n_times, attrs, double=False):
        self.path = Path(path)
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")
        self.kind = "f8" if double else "f4"
        self.source = source
        self.ds = Dataset(self.tmp_path, "w", format="NETCDF4")
        self.diag = self.ds.createGroup("diagnostics")
        self._define_grid(source, roi, n_times)
        self._define_attributes(source, roi, crop, attrs)
        self._fields_defined = False
        self._diagnostics_defined = False

    def _define_grid(self, source, roi, n_times):
        ds = self.ds
        self.time_dim = source.time_var.dimensions[0]
        ds.createDimension(self.time_dim, n_times)
        ds.createDimension(source.z_dim, source.grid.nz)
        ds.createDimension(source.zf_dim, source.grid.nz + 1)
        ds.createDimension(source.y_dim, roi.ny)
        ds.createDimension(source.x_dim, roi.nx)
        ds.createDimension(source.x_face_dim, roi.nx + 1)
        ds.createDimension(source.y_face_dim, roi.ny + 1)
        for coord, dim, values in (
            (source.x_center, source.x_dim, source.x_center[roi.i0 : roi.i1]),
            (source.y_center, source.y_dim, source.y_center[roi.j0 : roi.j1]),
            (source.x_face, source.x_face_dim, source.x_face[roi.i0 : roi.i1 + 1]),
            (source.y_face, source.y_face_dim, source.y_face[roi.j0 : roi.j1 + 1]),
            (source.z_coord, source.z_dim, source.z_coord[:]),
            (source.zf_coord, source.zf_dim, source.zf_coord[:]),
        ):
            var = ds.createVariable(coord.name, "f8", (dim,))
            var[:] = np.asarray(values, dtype=np.float64)
            var.units = coord.units
        self.time_out = ds.createVariable(source.time_var.name, "f8", (self.time_dim,))
        self.time_out.units = getattr(source.time_var, "units", "seconds")
        self.diag.createDimension("crop", len(vforce.CROP_FRACTIONS))
        fractions = self.diag.createVariable("crop_fraction", "f8", ("crop",))
        fractions[:] = vforce.CROP_FRACTIONS
        fractions.long_name = "fraction of the buffer trimmed from every side"

        profiles = {
            "th0": (source.z_dim, "K", "base-state potential temperature"),
            "qv0": (source.z_dim, "kg kg-1", "base-state water vapor mixing ratio"),
            "p0": (source.z_dim, "Pa", "base-state pressure"),
            "rho0": (source.z_dim, "kg m-3", "base-state density used in the decomposition"),
            "rho0_f": (source.zf_dim, "kg m-3", "base-state density at z-faces"),
        }
        for name, (dim, units, long_name) in profiles.items():
            var = ds.createVariable(name, "f8", (self.time_dim, dim))
            var.units, var.long_name = units, long_name
        if "cref" in source.ds.variables:
            var = ds.createVariable("cref", "f4", (self.time_dim, source.y_dim, source.x_dim), **COMPRESSION)
            var.units, var.long_name = "dBZ", "composite reflectivity"

    def _define_attributes(self, source, roi, crop, attrs):
        for key in source.ds.ncattrs():
            self.ds.setncattr(key, source.ds.getncattr(key))
        self.ds.setncatts(
            {
                "method": "Xu et al. (2015a) DVPGF / effective-buoyancy split, KR83 boundary conditions",
                "source_file": str(source.path),
                "dx_m": source.dx,
                "dy_m": source.dy,
                "roi_i0": roi.i0,
                "roi_i1": roi.i1,
                "roi_j0": roi.j0,
                "roi_j1": roi.j1,
                "crop_i0": crop.i0,
                "crop_i1": crop.i1,
                "crop_j0": crop.j0,
                "crop_j1": crop.j1,
                "hydrometeors": ",".join(source.hydrometeors),
                "field_precision": self.kind,
                **attrs,
            }
        )

    def _define_fields(self, fields, variant):
        source = self.source
        dims = {
            "center": (self.time_dim, source.z_dim, source.y_dim, source.x_dim),
            "face": (self.time_dim, source.zf_dim, source.y_dim, source.x_dim),
            "xface": (self.time_dim, source.z_dim, source.y_dim, source.x_face_dim),
            "yface": (self.time_dim, source.z_dim, source.y_face_dim, source.x_dim),
        }
        for name, data in fields.items():
            location, units, long_name = FIELD_ATTRS[name]
            if variant == "H" and name in ("p_d", "p_b"):
                units, long_name = "1", long_name.replace("pressure", "Exner") + " (pi')"
            chunks = (1, 1) + data.shape[1:]
            var = self.ds.createVariable(name, self.kind, dims[location], chunksizes=chunks, **COMPRESSION)
            var.units, var.long_name = units, long_name
        self._fields_defined = True

    def _define_diagnostics(self, values):
        for name, value in values.items():
            array = np.asarray(value)
            kind = "i4" if np.issubdtype(array.dtype, np.integer) else "f8"
            dims = (self.time_dim, "crop") if array.ndim == 1 else (self.time_dim,)
            var = self.diag.createVariable(name, kind, dims)
            units, long_name = DIAGNOSTIC_ATTRS.get(name, ("1", name.replace("_", " ")))
            var.units, var.long_name = units, long_name
        self._diagnostics_defined = True

    def write(self, index, time, fields, base, cref, diagnostics, variant):
        self.time_out[index] = time
        if not self._fields_defined:
            self._define_fields(fields, variant)
        for name, data in fields.items():
            self.ds[name][index] = data
        for name, values in (
            ("th0", base.th),
            ("qv0", base.qv),
            ("p0", base.p),
            ("rho0", base.rho),
            ("rho0_f", base.rho_f),
        ):
            self.ds[name][index] = values
        if cref is not None and "cref" in self.ds.variables:
            self.ds["cref"][index] = cref
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

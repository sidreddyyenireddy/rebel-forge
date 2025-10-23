#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYSCF_SRC = ROOT / "pyscf"
SKALA_SRC = ROOT / "skala" / "src"
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Ensure Skala sources are importable regardless of PySCF resolution order
sys.path.insert(0, str(SKALA_SRC))


def _ensure_pyscf_shared_libs() -> bool:
    """Populate local PySCF lib directory with prebuilt shared libraries if available."""

    local_lib_dir = PYSCF_SRC / "pyscf" / "lib"
    if not local_lib_dir.exists():
        return False

    def _has_libxc_binary(directory: Path) -> bool:
        for ext in ("*.so", "*.dylib", "*.dll"):
            if any(directory.glob(f"libxc{ext}")):
                return True
        return False

    def _has_dependency_libs(directory: Path) -> bool:
        deps_dir = directory / "deps" / "lib"
        required = ["libcint", "libxc", "libxcfun"]
        if not deps_dir.exists():
            return False
        for name in required:
            if not any(deps_dir.glob(f"{name}*")):
                return False
        return True

    if _has_libxc_binary(local_lib_dir) and _has_dependency_libs(local_lib_dir):
        return True

    original_path = list(sys.path)
    try:
        sys.path = [p for p in original_path if p not in ('', str(ROOT), str(PYSCF_SRC))]
        spec = importlib.util.find_spec("pyscf")
    finally:
        sys.path = original_path
    if not spec or not spec.submodule_search_locations:
        return False

    external_lib_dir = Path(spec.submodule_search_locations[0]) / "lib"
    if not external_lib_dir.exists():
        return False

    for candidate in external_lib_dir.glob("lib*" ):
        target = local_lib_dir / candidate.name
        if target.exists():
            continue
        try:
            shutil.copy2(candidate, target)
            print(f"[run_dft] Copied {candidate.name} to local PySCF lib", file=sys.stderr)
        except OSError as err:
            print(f"[run_dft] Failed to copy {candidate}: {err}", file=sys.stderr)

    deps_src = external_lib_dir / "deps"
    deps_dst = local_lib_dir / "deps"
    if deps_src.exists() and not deps_dst.exists():
        try:
            shutil.copytree(deps_src, deps_dst)
            print("[run_dft] Copied PySCF dependency libraries", file=sys.stderr)
        except OSError as err:
            print(f"[run_dft] Failed to copy dependency libraries: {err}", file=sys.stderr)

    return _has_libxc_binary(local_lib_dir) and _has_dependency_libs(local_lib_dir)


USE_LOCAL_PYSCF = os.environ.get("RF_USE_LOCAL_PYSCF", "0") in ("1", "true", "True")
if USE_LOCAL_PYSCF:
    # Only attempt to use the vendored PySCF checkout if explicitly requested
    USE_LOCAL_PYSCF = _ensure_pyscf_shared_libs()

if not USE_LOCAL_PYSCF:
    # Always ensure a fresh pip-installed PySCF so users get an up-to-date copy
    try:
        print("[run_dft] Ensuring PySCF (pip, latest)…", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pyscf"])
    except Exception as _e:  # pylint: disable=broad-except
        # Proceed even if upgrade fails; import may still succeed with an existing install
        print(f"[run_dft] PySCF upgrade failed ({_e}); proceeding with existing install if available.", file=sys.stderr)

if USE_LOCAL_PYSCF:
    try:
        import ctypes

        lib_dir = PYSCF_SRC / "pyscf" / "lib"
        # Pick platform-appropriate extension; fallback to any available match
        ext = "dll" if sys.platform.startswith("win") else ("dylib" if sys.platform == "darwin" else "so")
        lib_path = lib_dir / f"libxc_itrf.{ext}"
        if not lib_path.exists():
            candidates = sorted(lib_dir.glob("libxc_itrf.*"))
            if candidates:
                lib_path = candidates[0]
        lib = ctypes.CDLL(str(lib_path))
        getattr(lib, "LIBXC_xc_func_init")
    except Exception as err:  # pylint: disable=broad-except
        print(f"[run_dft] Local PySCF binaries missing required symbols ({err}); falling back to system PySCF.", file=sys.stderr)
        USE_LOCAL_PYSCF = False

if USE_LOCAL_PYSCF:
    # Force local PySCF checkout to be used, now that it has the required libraries
    sys.path.insert(0, str(PYSCF_SRC))
else:
    # Remove the incomplete local checkout from the search path so the system PySCF is used
    local_entry = str(PYSCF_SRC)
    root_entry = str(ROOT)
    sys.path = [p for p in sys.path if p not in ('', root_entry, local_entry)]

# Ensure dftd3 dependency is available for Skala integration
try:
    import dftd3  # type: ignore  # pylint: disable=unused-import
except ModuleNotFoundError:
    print("[run_dft] Installing dftd3…", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dftd3"])

from pyscf import gto  # type: ignore  # pylint: disable=wrong-import-position
import pyscf
PRINTED_PYSCF_PATH = getattr(pyscf, '__file__', 'unknown')
print(f"[run_dft] Using PySCF from {PRINTED_PYSCF_PATH}", file=sys.stderr)
from skala.pyscf import SkalaKS  # type: ignore  # pylint: disable=wrong-import-position


def load_xyz(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return []
    try:
        atom_count = int(lines[0])
        start_idx = 2 if len(lines) > 1 else 1
        candidates = lines[start_idx : start_idx + atom_count]
    except ValueError:
        candidates = lines
    atoms: list[str] = []
    for line in candidates:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            continue
        atoms.append(f"{symbol} {x} {y} {z}")
    return atoms


def load_pdb(path: Path) -> list[str]:
    atoms: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            symbol = line[76:78].strip() or line[12:16].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atoms.append(f"{symbol} {x} {y} {z}")
    return atoms


def load_pyscf_atom(path: Path) -> list[str]:
    """Load PySCF `atom` text.

    Accepts either:
      - One atom per line: "El x y z"
      - Semicolon-separated atoms on one or more lines: "El x y z; El x y z; ..."
    """
    text = path.read_text(encoding="utf-8")
    # Normalize separators: turn semicolons into newlines
    normalized = "\n".join(part.strip() for part in text.replace(";", "\n").splitlines())
    atoms: list[str] = []
    for raw in normalized.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            continue
        atoms.append(f"{symbol} {x} {y} {z}")
    return atoms


def load_sdf_mol(path: Path) -> list[str]:
    """Parse MDL MOL/SDF (V2000/V3000) and return 'El x y z' list."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    # Check for V3000 first
    if any("V3000" in ln for ln in lines[:20]):
        nat = 0
        atoms: list[str] = []
        in_atom = False
        for ln in lines:
            if ln.startswith("M  V30 ") and "COUNTS" in ln:
                parts = ln.split()
                try:
                    idx = parts.index("COUNTS")
                    nat = int(parts[idx + 1])
                except Exception:
                    nat = 0
            elif ln.startswith("M  V30 BEGIN ATOM"):
                in_atom = True
            elif ln.startswith("M  V30 END ATOM"):
                in_atom = False
            elif in_atom and ln.startswith("M  V30"):
                parts = ln.split()
                if len(parts) >= 7:
                    sym = parts[3]
                    try:
                        x = float(parts[4]); y = float(parts[5]); z = float(parts[6])
                    except ValueError:
                        continue
                    atoms.append(f"{sym} {x} {y} {z}")
                    if nat and len(atoms) >= nat:
                        break
        return atoms

    # V2000 fallback
    if len(lines) < 4:
        return []
    counts = lines[3]
    nat = 0
    try:
        nat = int(counts[0:3])
    except Exception:
        parts = counts.split()
        for tok in parts:
            if tok.isdigit():
                nat = int(tok)
                break
    atoms: list[str] = []
    for ln in lines[4 : 4 + max(nat, 0)]:
        s = ln.rstrip()
        if not s:
            continue
        try:
            x = float(s[0:10]); y = float(s[10:20]); z = float(s[20:30]); sym = s[31:34].strip()
        except Exception:
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = map(float, parts[:3])
            except ValueError:
                continue
            sym = parts[3]
        atoms.append(f"{sym} {x} {y} {z}")
    return atoms


def load_mol2(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    atoms: list[str] = []
    in_atom = False
    for ln in text.splitlines():
        s = ln.strip()
        ls = s.upper()
        if ls.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if ls.startswith("@<TRIPOS>BOND"):
            break
        if in_atom:
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
            except ValueError:
                continue
            atom_type = parts[5]
            sym = atom_type.split('.')[0]
            atoms.append(f"{sym} {x} {y} {z}")
    return atoms


def load_cml(path: Path) -> list[str]:
    atoms: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return atoms
    for el in root.iter():
        if el.tag.endswith('atom'):
            sym = el.get('elementType') or el.get('element') or el.get('el')
            try:
                x = float(el.get('x3') or el.get('x') or el.get('x2') or 'nan')
                y = float(el.get('y3') or el.get('y') or el.get('y2') or 'nan')
                z = float(el.get('z3') or el.get('z') or '0.0')
            except ValueError:
                continue
            if sym and (not math.isnan(x)) and (not math.isnan(y)) and (not math.isnan(z)):
                atoms.append(f"{sym} {x} {y} {z}")
    return atoms


def _lattice_from_cif(a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float):
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * math.cos(gamma), b * math.sin(gamma), 0.0
    cx = c * math.cos(beta)
    sin_g = math.sin(gamma) if abs(math.sin(gamma)) > 1e-8 else 1e-8
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_g
    cz_sq = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(0.0, cz_sq))
    return (ax, ay, az), (bx, by, bz), (cx, cy, cz)


def _load_cif_with_gemmi(path: Path) -> list[str] | None:
    try:
        import gemmi  # type: ignore
    except ModuleNotFoundError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gemmi"])
        except Exception as err:  # pylint: disable=broad-except
            print(f"[run_dft] Warning: unable to install gemmi ({err}). Falling back to manual CIF parser.", file=sys.stderr)
            return None
        import gemmi  # type: ignore  # pylint: disable=import-error
    try:
        doc = gemmi.cif.read_file(str(path))
    except Exception as err:  # pylint: disable=broad-except
        print(f"[run_dft] Warning: gemmi could not read CIF file ({err}).", file=sys.stderr)
        return None
    if not doc or not getattr(doc, "blocks", None):
        return None
    try:
        block = doc.sole_block() if hasattr(doc, "sole_block") else doc[0]
    except Exception:  # pylint: disable=broad-except
        block = doc[0] if len(doc) else None
    if block is None:
        return None
    try:
        structure = gemmi.make_structure_from_block(block)
    except Exception as err:  # pylint: disable=broad-except
        print(f"[run_dft] Warning: gemmi failed to build structure ({err}).", file=sys.stderr)
        return None
    try:
        structure.remove_alternative_conformations()
    except Exception:
        pass
    atoms: list[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    element = atom.element.name.strip()
                    if not element or element.upper() in ('.', '?', 'X'):
                        continue
                    pos = atom.pos
                    atoms.append(f"{element} {pos.x:.8f} {pos.y:.8f} {pos.z:.8f}")
    return atoms or None


def _load_cif_manual(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    a = b = c = 0.0
    alpha = beta = gamma = 90.0
    for ln in lines[:200]:
        low = ln.lower()
        try:
            if low.startswith('_cell_length_a'):
                a = float(ln.split()[1])
            elif low.startswith('_cell_length_b'):
                b = float(ln.split()[1])
            elif low.startswith('_cell_length_c'):
                c = float(ln.split()[1])
            elif low.startswith('_cell_angle_alpha'):
                alpha = float(ln.split()[1])
            elif low.startswith('_cell_angle_beta'):
                beta = float(ln.split()[1])
            elif low.startswith('_cell_angle_gamma'):
                gamma = float(ln.split()[1])
        except Exception:
            pass
    atoms: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].lower() == 'loop_':
            i += 1
            cols = []
            while i < len(lines) and lines[i].startswith('_'):
                cols.append(lines[i]); i += 1
            normalized_cols = [c.strip().lower().replace('.', '_') for c in cols]
            column_map = {name: idx for idx, name in enumerate(normalized_cols)}
            coord_mode = None
            sx = sy = sz = -1
            loop_family = ''
            coord_candidates = [
                (('_atom_site_cartn_x', '_atom_site_cartn_y', '_atom_site_cartn_z'), 'cartn', '_atom_site'),
                (('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z'), 'fract', '_atom_site'),
                (('_chem_comp_atom_model_cartn_x', '_chem_comp_atom_model_cartn_y', '_chem_comp_atom_model_cartn_z'), 'cartn', '_chem_comp_atom'),
                (('_chem_comp_atom_pdbx_model_cartn_x_ideal', '_chem_comp_atom_pdbx_model_cartn_y_ideal', '_chem_comp_atom_pdbx_model_cartn_z_ideal'), 'cartn', '_chem_comp_atom'),
                (('_chem_comp_atom_model_fract_x', '_chem_comp_atom_model_fract_y', '_chem_comp_atom_model_fract_z'), 'fract', '_chem_comp_atom'),
            ]
            for keys, mode, family in coord_candidates:
                if all(key in column_map for key in keys):
                    sx, sy, sz = (column_map[key] for key in keys)
                    coord_mode = mode
                    loop_family = family
                    break
            if coord_mode is None:
                continue
            group_idx = column_map.get('_atom_site_group_pdb')
            group_default = 'HETATM' if loop_family == '_chem_comp_atom' else ''
            sym_candidates = [
                '_atom_site_type_symbol',
                '_atom_site_label_atom_id',
                '_atom_site_auth_atom_id',
                '_atom_site_label',
                '_chem_comp_atom_type_symbol',
                '_chem_comp_atom_atom_id',
                '_chem_comp_atom_alt_atom_id',
                '_chem_comp_atom_pdbx_component_atom_id',
            ]
            ss = next((column_map[name] for name in sym_candidates if name in column_map), sx)
            symbol_source = normalized_cols[ss] if ss < len(normalized_cols) else ''
            while i < len(lines) and not lines[i].lower().startswith(('loop_', 'data_', 'global_', '_')):
                parts = lines[i].split()
                if len(parts) <= max(sx, sy, sz, ss):
                    i += 1; continue
                group_value = ""
                if group_idx is not None and group_idx < len(parts):
                    group_value = parts[group_idx]
                elif group_default:
                    group_value = group_default
                sym = _normalize_cif_symbol(parts[ss], symbol_source, group_value)
                if not sym:
                    i += 1; continue
                try:
                    raw_x, raw_y, raw_z = parts[sx], parts[sy], parts[sz]
                    if raw_x in ('.', '?') or raw_y in ('.', '?') or raw_z in ('.', '?'):
                        i += 1; continue
                    x = float(raw_x); y = float(raw_y); z = float(raw_z)
                except ValueError:
                    i += 1; continue
                if coord_mode == 'fract':
                    a_vec, b_vec, c_vec = _lattice_from_cif(a, b, c, alpha, beta, gamma)
                    cx = x * a_vec[0] + y * b_vec[0] + z * c_vec[0]
                    cy = x * a_vec[1] + y * b_vec[1] + z * c_vec[1]
                    cz = x * a_vec[2] + y * b_vec[2] + z * c_vec[2]
                    x, y, z = cx, cy, cz
                atoms.append(f"{sym} {x} {y} {z}")
                i += 1
            continue
        i += 1
    return atoms


def load_cif(path: Path) -> list[str]:
    atoms = _load_cif_with_gemmi(path)
    if atoms:
        return atoms
    return _load_cif_manual(path)


def parse_structure(structure_path: Path, fmt: str) -> list[str]:
    fmt = fmt.lower()
    if fmt == "xyz":
        atoms = load_xyz(structure_path)
    elif fmt == "pdb":
        atoms = load_pdb(structure_path)
    elif fmt == "pyscf":
        atoms = load_pyscf_atom(structure_path)
    elif fmt in ("sdf", "sd", "mol"):
        atoms = load_sdf_mol(structure_path)
    elif fmt == "mol2":
        atoms = load_mol2(structure_path)
    elif fmt == "cml":
        atoms = load_cml(structure_path)
    elif fmt in ("cif", "mmcif"):
        atoms = load_cif(structure_path)
    else:
        raise ValueError(f"Unsupported structure format: {fmt}")
    if not atoms:
        raise ValueError("No atoms could be parsed from the supplied structure.")
    return atoms


def persist_pyscf_atoms(atoms: list[str], source_path: Path) -> Path | None:
    """Write parsed atoms to a .pyscf text file next to the source."""
    if not atoms:
        return None
    if source_path.suffix.lower() == ".pyscf":
        return source_path
    try:
        target = source_path.with_suffix(".pyscf")
    except ValueError:
        target = source_path.parent / (source_path.name + ".pyscf")
    try:
        text = "\n".join(atoms) + "\n"
        target.write_text(text, encoding="utf-8")
        return target
    except OSError as err:
        print(f"[run_dft] Warning: unable to persist PySCF atoms to {target}: {err}", file=sys.stderr)
        return None


def _std_symbol(sym: str) -> str:
    if not sym:
        return sym
    return sym[0].upper() + sym[1:].lower()


_ELEMENTS_ORDER = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm "
    "Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No "
    "Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_SYMBOL_TO_Z: dict[str, int] = {sym: idx for idx, sym in enumerate(_ELEMENTS_ORDER) if idx > 0}
_AMBIGUOUS_PDB_LABELS = {
    "CA", "CB", "CG", "CD", "CE", "CZ", "CH",
    "ND", "NE", "NZ", "NH",
    "OD", "OE", "OG", "OH",
    "SD", "SG",
    "HA", "HB", "HG", "HD", "HE", "HZ", "HH",
}


def _infer_charge_and_spin(atoms: list[str]) -> tuple[int, int]:
    """Infer a neutral charge and a spin consistent with electron parity.

    - Default to neutral (charge=0)
    - Compute Z sum from element symbols
    - Set spin = (Z_sum - charge) % 2 to satisfy PySCF parity constraint
    """
    z_sum = 0
    for line in atoms:
        symbol = _std_symbol(line.split()[0])
        z = _SYMBOL_TO_Z.get(symbol, 0)
        z_sum += z
    charge = 0
    ne = z_sum - charge
    spin = int(ne % 2)
    return charge, spin


def _normalize_cif_symbol(token: str, symbol_source: str, group_pdb: str) -> str:
    raw = token.strip().strip("\"'")
    if not raw or raw in ('.', '?'):
        return ''
    letters = ''.join(ch for ch in raw if ch.isalpha())
    if not letters:
        return ''
    letters_up = letters[:2].upper()
    prefer_single = symbol_source != '_atom_site_type_symbol'
    if letters_up in _AMBIGUOUS_PDB_LABELS and group_pdb.upper() != 'HETATM':
        prefer_single = True
    if len(letters) >= 2 and not prefer_single:
        candidate = letters_up[0] + letters_up[1].lower()
        if candidate in _SYMBOL_TO_Z:
            return candidate
    first = letters[0].upper()
    return first if first in _SYMBOL_TO_Z else ''


def _build_mol_with_defaults(atoms: list[str]) -> tuple["gto.Mole", str, int, int]:  # type: ignore[name-defined]
    charge, spin = _infer_charge_and_spin(atoms)
    basis_order = [os.environ.get("RF_DEFAULT_BASIS", "def2-svp"), "sto-3g"]
    last_err: Exception | None = None
    for basis in basis_order:
        try:
            mol = gto.M(atom="; ".join(atoms), unit="Angstrom", basis=basis, charge=charge, spin=spin)
            return mol, basis, charge, spin
        except Exception as err:  # pylint: disable=broad-except
            last_err = err
            continue
    assert last_err is not None
    raise last_err


def _find_model_path() -> Path:
    """Resolve location for skala-1.0.fun within this checkout.

    Search priority:
      1. SKALA_LOCAL_MODEL_PATH env var
      2. ./skala/models/skala-1.0.fun
      3. ./skala/skala-1.0.fun
      4. ./skala-1.0.fun (repo root)
    """
    model_path_env = os.environ.get("SKALA_LOCAL_MODEL_PATH")
    if model_path_env:
        p = Path(model_path_env)
        if p.exists():
            return p
    candidates = [
        ROOT / "skala" / "models" / "skala-1.0.fun",
        ROOT / "skala" / "skala-1.0.fun",
        ROOT / "skala-1.0.fun",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Default (historical) location under skala/models
    return candidates[0]


def _coords_angstrom(mol):
    try:
        return mol.atom_coords(unit="Angstrom")
    except TypeError:
        import numpy as _np

        bohr = 0.529177210903
        return _np.asarray(mol.atom_coords()) * bohr


def _save_xyz(path: Path, mol) -> Path:
    coords = _coords_angstrom(mol)
    lines = [str(mol.natm), "optimized geometry (Angstrom)"]
    for idx in range(mol.natm):
        sym = mol.atom_symbol(idx)
        x, y, z = coords[idx]
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_skala_with_current_settings(mol, fast_mode: bool):
    ks = SkalaKS(mol, xc="skala", with_dftd3=False, with_density_fit=False)
    try:
        grid_level_env = os.environ.get("RF_GRID_LEVEL")
        if grid_level_env is not None:
            ks.grids.level = int(grid_level_env)
        elif fast_mode:
            ks.grids.level = 1
        atom_grid_env = os.environ.get("RF_ATOM_GRID")
        if atom_grid_env:
            try:
                radial, angular = atom_grid_env.lower().replace("x", " ").split()
                ks.grids.atom_grid = (int(radial), int(angular))
            except Exception:  # pylint: disable=broad-except
                pass
        elif fast_mode:
            ks.grids.atom_grid = (30, 110)
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        if os.environ.get("RF_MAX_CYCLE"):
            ks.max_cycle = int(os.environ["RF_MAX_CYCLE"])
        elif fast_mode:
            ks.max_cycle = 50
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        if os.environ.get("RF_CONV_TOL"):
            ks.conv_tol = float(os.environ["RF_CONV_TOL"])
        elif fast_mode:
            ks.conv_tol = 1e-6
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        # Ensure PySCF emits progress logs; allow override with RF_VERBOSE
        ks.verbose = int(os.environ.get("RF_VERBOSE", "4"))
    except Exception:  # pylint: disable=broad-except
        pass
    return ks


def _compute_forces(ks):
    gradients = ks.nuc_grad_method().kernel()
    import numpy as _np

    forces = -_np.array(gradients)
    return gradients.tolist(), forces.tolist()


_GEOMETRIC_SOLVER = None
_GEOMETRIC_INITIALIZED = False
_GEOMETRIC_ERROR: str | None = None


def _ensure_geometric_solver():
    global _GEOMETRIC_SOLVER  # pylint: disable=global-statement
    global _GEOMETRIC_INITIALIZED  # pylint: disable=global-statement
    global _GEOMETRIC_ERROR  # pylint: disable=global-statement

    if _GEOMETRIC_INITIALIZED:
        return _GEOMETRIC_SOLVER, _GEOMETRIC_ERROR

    _GEOMETRIC_INITIALIZED = True
    try:
        from pyscf.geomopt import geometric_solver as solver  # type: ignore

        _GEOMETRIC_SOLVER = solver
        _GEOMETRIC_ERROR = None
        return _GEOMETRIC_SOLVER, None
    except Exception as import_err:  # pylint: disable=broad-except
        print(f"[run_dft] geometric solver import failed ({import_err}); attempting installation.", file=sys.stderr)
        try:
            print("[run_dft] Installing geometric…", file=sys.stderr)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "geometric"])
            from pyscf.geomopt import geometric_solver as solver  # type: ignore

            _GEOMETRIC_SOLVER = solver
            _GEOMETRIC_ERROR = None
            return _GEOMETRIC_SOLVER, None
        except Exception as install_err:  # pylint: disable=broad-except
            _GEOMETRIC_SOLVER = None
            _GEOMETRIC_ERROR = str(install_err)
            print(f"[run_dft] Warning: geometric optimizer unavailable ({install_err}).", file=sys.stderr)
            return None, _GEOMETRIC_ERROR


def run_dft(
    structure_path: Path,
    fmt: str,
    *,
    include_forces: bool = False,
    optimize_geometry: bool = False,
) -> dict:
    atoms = parse_structure(structure_path, fmt)
    pyscf_path = persist_pyscf_atoms(atoms, structure_path)
    
    fast_mode = os.environ.get("RF_FAST_MODE", "1") in ("1", "true", "True")
    if fast_mode:
        os.environ.setdefault("RF_DEFAULT_BASIS", "sto-3g")
    mol, basis_used, charge, spin = _build_mol_with_defaults(atoms)
    # Ensure molecule verbosity is not silenced
    try:
        mol.verbose = int(os.environ.get("RF_VERBOSE", "4"))  # type: ignore[attr-defined]
    except Exception:  # pylint: disable=broad-except
        pass

    model_path = _find_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            "Skala model file not found. Place 'skala-1.0.fun' under skala/models/ or set SKALA_LOCAL_MODEL_PATH to the file location."
        )
    os.environ["SKALA_LOCAL_MODEL_PATH"] = str(model_path)

    ks = _build_skala_with_current_settings(mol, fast_mode)
    # Honour RF_VERBOSE on the KS object too
    try:
        ks.verbose = int(os.environ.get("RF_VERBOSE", str(getattr(ks, "verbose", 4))))
    except Exception:  # pylint: disable=broad-except
        pass
    energy_initial = float(ks.kernel())
    try:
        converged_initial = bool(getattr(ks, "converged", None))
    except Exception:  # pylint: disable=broad-except
        converged_initial = None

    result: dict[str, object] = {
        "energy": energy_initial,
        "energy_initial": energy_initial,
        "natoms": len(atoms),
        "nelectron": int(mol.nelectron),
        "nmo": int(mol.nao_nr()),
        "basis": basis_used,
        "charge": charge,
        "spin": spin,
        "sequence_format": "pyscf",
        "sequence": "\n".join(atoms),
        "source_format": fmt,
        **({"pyscf_path": str(pyscf_path)} if pyscf_path else {}),
        "include_forces": bool(include_forces or optimize_geometry),
        "optimize_geometry": bool(optimize_geometry),
        "task": "geomopt" if optimize_geometry else "dft",
    }
    if converged_initial is not None:
        result["converged_initial"] = converged_initial

    forces_path: Path | None = None
    if include_forces or optimize_geometry:
        include_forces = True
    if include_forces:
        print("[run_dft] Computing gradients/forces…", file=sys.stderr)
        gradients, forces = _compute_forces(ks)
        forces_path = structure_path.parent / "forces.json"
        forces_payload = {
            "units": {"gradients": "Hartree/Bohr", "forces": "Hartree/Bohr"},
            "gradients": gradients,
            "forces": forces,
            "natoms": len(atoms),
        }
        try:
            forces_path.write_text(json.dumps(forces_payload, indent=2) + "\n", encoding="utf-8")
            result["forces_json"] = str(forces_path)
            result["forces_units"] = "Hartree/Bohr"
            result["gradients_units"] = "Hartree/Bohr"
        except OSError as err:
            print(f"[run_dft] Warning: unable to write forces JSON to {forces_path}: {err}", file=sys.stderr)

    optimized_xyz_path: Path | None = None
    if optimize_geometry:
        print("[run_dft] Running geometry optimization…", file=sys.stderr)
        solver, solver_error = _ensure_geometric_solver()
        if solver is None:
            message = solver_error or "geometric solver not installed"
            result["optimization_unavailable"] = f"Optimization unavailable: {message}"
        else:
            try:
                mol_opt = solver.optimize(ks)  # type: ignore[attr-defined]
            except Exception as opt_err:  # pylint: disable=broad-except
                print(f"[run_dft] Optimization failed: {opt_err}", file=sys.stderr)
                result["optimization_error"] = f"Optimization failed: {opt_err}"
                mol_opt = None
            if mol_opt is not None:
                optimized_xyz_path = structure_path.parent / "optimized.xyz"
                try:
                    _save_xyz(optimized_xyz_path, mol_opt)
                    result["optimized_xyz"] = str(optimized_xyz_path)
                except OSError as err:
                    print(f"[run_dft] Warning: unable to write optimized geometry to {optimized_xyz_path}: {err}", file=sys.stderr)

                ks_opt = _build_skala_with_current_settings(mol_opt, fast_mode)
                try:
                    ks_opt.verbose = int(os.environ.get("RF_VERBOSE", str(getattr(ks_opt, "verbose", 4))))
                except Exception:  # pylint: disable=broad-except
                    pass
                energy_final = float(ks_opt.kernel())
                result["energy_final"] = energy_final
                try:
                    result["converged"] = bool(getattr(ks_opt, "converged", None))
                except Exception:  # pylint: disable=broad-except
                    pass
                try:
                    result["nmo_final"] = int(mol_opt.nao_nr())
                    result["nelectron_final"] = int(mol_opt.nelectron)
                except Exception:  # pylint: disable=broad-except
                    pass

    output_path = structure_path.parent / "dft_result.json"
    try:
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["result_file"] = str(output_path)
        print(f"[run_dft] Results saved to {output_path}", file=sys.stderr)
    except OSError as err:
        print(f"[run_dft] Warning: unable to write result JSON to {output_path}: {err}", file=sys.stderr)
    return result


def _run_common_build(structure_path: Path, fmt: str):
    atoms = parse_structure(structure_path, fmt)
    pyscf_path = persist_pyscf_atoms(atoms, structure_path)

    fast_mode = os.environ.get("RF_FAST_MODE", "1") in ("1", "true", "True")
    if fast_mode:
        os.environ.setdefault("RF_DEFAULT_BASIS", "sto-3g")
    mol, basis_used, charge, spin = _build_mol_with_defaults(atoms)
    try:
        mol.verbose = int(os.environ.get("RF_VERBOSE", "4"))  # type: ignore[attr-defined]
    except Exception:  # pylint: disable=broad-except
        pass

    model_path = _find_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            "Skala model file not found. Place 'skala-1.0.fun' under skala/models/ or set SKALA_LOCAL_MODEL_PATH to the file location."
        )
    os.environ["SKALA_LOCAL_MODEL_PATH"] = str(model_path)

    ks = _build_skala_with_current_settings(mol, fast_mode)
    try:
        ks.verbose = int(os.environ.get("RF_VERBOSE", str(getattr(ks, "verbose", 4))))
    except Exception:  # pylint: disable=broad-except
        pass

    meta: dict[str, object] = {
        "natoms": len(atoms),
        "nelectron": int(mol.nelectron),
        "nmo": int(mol.nao_nr()),
        "basis": basis_used,
        "charge": charge,
        "spin": spin,
        "sequence_format": "pyscf",
        "sequence": "\n".join(atoms),
        "source_format": fmt,
    }
    if pyscf_path:
        meta["pyscf_path"] = str(pyscf_path)
    return mol, ks, meta




def main() -> None:
    parser = argparse.ArgumentParser(description="Run PySCF DFT with the Skala functional.")
    parser.add_argument("--structure", required=True, help="Path to structure file.")
    parser.add_argument(
        "--format",
        default="xyz",
        choices=("xyz", "pdb", "pyscf", "sdf", "sd", "mol", "mol2", "cml", "cif"),
        help="Structure format.",
    )
    parser.add_argument(
        "--task",
        choices=("dft", "geomopt"),
        default="dft",
        help="DFT only or geometry optimisation with forces.",
    )
    parser.add_argument(
        "--include-forces",
        action="store_true",
        dest="include_forces",
        help="Compute gradients/forces on the initial geometry.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        dest="optimize",
        help="Run geometry optimisation (implies --include-forces).",
    )
    parser.add_argument("--label", default="", help="Optional label for logging.")
    args = parser.parse_args()

    structure_path = Path(args.structure).resolve()
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    task = args.task
    optimize_geometry = bool(args.optimize or task == "geomopt")
    include_forces = bool(args.include_forces or optimize_geometry)
    result = run_dft(
        structure_path,
        args.format,
        include_forces=include_forces,
        optimize_geometry=optimize_geometry,
    )
    if args.label:
        result["label"] = args.label
    print(json.dumps(result))


if __name__ == "__main__":
    main()

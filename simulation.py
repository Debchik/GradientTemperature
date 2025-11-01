"""
Simulation of particles in a 2D box with temperature‑dependent walls.

This module defines a Simulation class that models the motion of many
identical spherical particles in a unit square.  The particles move
ballistically until they collide with one another or the box walls.  When
they hit the top or bottom walls they reflect elastically.  When they
hit the left or right walls, their velocities are partially or fully
thermalized according to a user‑specified wall temperature and a
dimensionless accommodation coefficient.  Thermalization uses the
Maxwell–Smoluchowski distribution appropriate for 2D systems.

Unlike the original version of this project, there are no "spring"
particles linked by a harmonic potential.  All particles are free
entities.  Functions and state pertaining to the spring particles and
their potential energy have been removed.  To update wall temperatures
or the accommodation coefficient at run time, call
``Simulation.set_params`` with the appropriate keywords.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy import stats

################################################################################
# Thermal wall utility functions
################################################################################

# Boltzmann constant (scaled to simulation units).  Together with the
# particle mass this sets the characteristic thermal speed.  The value
# here is tuned so that wall temperatures of a few hundred kelvin
# produce noticeable motion for particles whose masses are taken from
# ``config.json``.
kB: float = 3.0e-6

# Base integration step in simulation time units.  Larger values speed up
# motion on screen but may slightly increase numerical error.
TIME_STEP: float = 6.0e-4

# Create a default random number generator.  If NumPy is unavailable
# ``rng`` will remain ``None`` and fallbacks will be used instead.
try:
    rng: np.random.Generator = np.random.default_rng()
except Exception:
    rng = None

def _gauss0(std: Union[float, np.ndarray], size=None):
    """Draw samples from N(0, std) for either scalar or array ``std``.

    Parameters
    ----------
    std: float or ndarray
        Standard deviation of the distribution.  If an array, it must be
        broadcastable to ``size``.
    size: int or tuple of int, optional
        Number of samples to draw when ``std`` is scalar.  Ignored if
        ``std`` is an array.

    Returns
    -------
    ndarray or float
        Normally distributed samples.
    """
    if rng is not None:
        return rng.normal(0.0, std, size=size)
    # Fallback to Box–Muller for scalar sampling
    if size is not None:
        raise RuntimeError("Vectorized sampling requires NumPy")
    u1 = max(1e-12, random.random())
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return std * z

def _rayleigh(sigma: Union[float, np.ndarray], size=None):
    """Draw samples from a Rayleigh distribution with scale parameter ``sigma``.

    When ``sigma`` is an array, a vector of Rayleigh samples of the same
    shape is returned.  Requires NumPy.
    """
    if rng is not None:
        if isinstance(sigma, np.ndarray) and size is None:
            # Generate elementwise Rayleigh samples via two normals
            z1 = rng.normal(0.0, sigma)
            z2 = rng.normal(0.0, sigma)
            return np.sqrt(z1 * z1 + z2 * z2)
        return rng.rayleigh(scale=sigma, size=size)
    raise RuntimeError("Vectorized Rayleigh sampling requires NumPy")

def sample_velocity_from_wall(T: float, masses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sample normal and tangential velocity components for diffusive reflection.

    Parameters
    ----------
    T: float
        Temperature of the wall in kelvin.
    masses: ndarray
        Array of particle masses (one per reflected particle).

    Returns
    -------
    v_n: ndarray
        The normal (outward) velocity component for each particle.
    v_t: ndarray
        The tangential velocity component for each particle.

    Notes
    -----
    In a 2D gas the distribution of speeds of molecules leaving a wall
    follows a Maxwell–Smoluchowski distribution: the normal component
    follows a half‑Maxwellian distribution proportional to v
    ``exp(-m v^2/(2 kT))``, and the tangential component is Gaussian
    ``N(0, kT/m)``.
    """
    sigma2 = (kB * T) / masses
    sigma = np.sqrt(sigma2)
    v_n = _rayleigh(sigma)
    v_t = _gauss0(sigma)
    return v_n, v_t

def reflect_with_accommodation(
    vx: np.ndarray,
    vy: np.ndarray,
    side: str,
    T_wall: float,
    accommodation: float,
    masses: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Apply thermal reflection rules when particles collide with a wall.

    Only the elements selected by ``mask`` are modified.  For the top and
    bottom walls a specular reflection is performed.  For the left and
    right walls, the tangential component follows the specular law while
    the normal component is blended between the specular reflection and a
    thermal velocity sampled from the Maxwell–Smoluchowski distribution
    at the wall temperature.

    Parameters
    ----------
    vx, vy: ndarray
        Arrays of x‑ and y‑velocity components for all particles.
    side: str
        One of ``'left'``, ``'right'``, ``'top'`` or ``'bottom'``.
    T_wall: float or None
        Temperature of the wall for diffusive reflection; ignored for
        top and bottom walls.
    accommodation: float
        Accommodation coefficient between 0 and 1.  A value of 1 means
        the outgoing velocity is fully thermalized; 0 means purely
        specular reflection.
    masses: ndarray
        Masses of the particles.
    mask: ndarray of bool
        Boolean mask selecting which particles are touching the wall and
        moving toward it.  Only these particles are updated.
    """
    if not np.any(mask):
        return

    a = float(np.clip(accommodation, 0.0, 1.0))

    if side == 'top' or side == 'bottom':
        # Purely specular: flip the sign of the y‑component
        vy[mask] = -vy[mask]
        return

    # For left and right walls we mix incoming and thermal components.
    # Local normal is +x for left wall and -x for right wall.
    if side == 'left':
        v_n_in = -vx[mask]  # incoming normal (>0 toward wall)
        v_spec = v_n_in  # specular reflection would give +v_n_in
        v_n_new, _ = sample_velocity_from_wall(T_wall, masses[mask])
        v_n_out = (1.0 - a) * v_spec + a * v_n_new
        # Ensure outward motion (positive vx)
        vx[mask] = np.abs(v_n_out)
        # Tangential component follows specular reflection (no change)
    elif side == 'right':
        v_n_in = vx[mask]
        v_spec = -v_n_in  # specular reflection would give -v_n_in
        v_n_new, _ = sample_velocity_from_wall(T_wall, masses[mask])
        v_n_out = (1.0 - a) * v_spec - a * v_n_new
        # Ensure outward motion (negative vx)
        vx[mask] = -np.abs(v_n_out)
        # Tangential component follows specular reflection (no change)


@dataclass
class ThermalWallConfig:
    """Dataclass storing thermal wall parameters.

    Attributes
    ----------
    T_left, T_right: float
        Temperatures (K) of the left and right walls.  A higher temperature
        results in particles leaving the wall with higher average speed.
    accommodation: float
        Accommodation coefficient controlling the mix of specular and
        diffusive reflection.  ``accommodation=1`` yields a fully
        diffusive (thermal) reflection; ``0`` yields a purely specular
        reflection.
    """
    T_left: float = 600.0
    T_right: float = 300.0
    accommodation: float = 1.0


thermal_cfg = ThermalWallConfig()

################################################################################
# Simulation class
################################################################################

class Simulation:
    """Evolve a gas of particles in a box with thermal walls.

    The simulation uses a simple elastic collision model between
    particles.  Particle positions and velocities are stored in
    continuous arrays for efficiency.  At each time step particles may
    collide with one another or the walls.  Left and right wall
    collisions thermalize velocities according to the specified wall
    temperatures and accommodation coefficient.  Top and bottom wall
    collisions reflect particles specularly.  No external potentials or
    springs are present in this variant.
    """

    def __init__(
        self,
        gamma: float,
        k: float,
        l_0: float,
        R: float,
        particles_cnt: int,
        T: float,
        m: ndarray,
    ):
        """Create a simulation with the given parameters.

        Parameters
        ----------
        gamma, k, l_0: float
            Legacy parameters from the original model (spring potential),
            unused in this version but accepted for API compatibility.
        R: float
            Radius of each particle in box units.  The simulation assumes
            the box extends from 0 to 1 in both x and y directions, so
            ``R`` must satisfy ``0 < R < 0.5``.
        particles_cnt: int
            Number of gas particles to simulate.
        T: float
            Initial gas temperature (K).  Velocities are drawn from
            Maxwellian distributions with this temperature.
        m: ndarray
            Masses of the particles (length ``particles_cnt``).  If a
            scalar is provided, it will be broadcast to the required
            shape.  Masses should be provided in SI units consistent
            with ``kB``.
        """
        # Store constants and parameters
        self._k_boltz: float = kB
        self._gamma: float = gamma
        self._k: float = k
        self._l_0: float = l_0
        self._R: float = R

        # Number of gas particles
        self._n_particles: int = int(particles_cnt)
        self._n_spring: int = 0  # no spring particles

        # Ensure masses have correct shape
        masses = np.asarray(m, dtype=float)
        if masses.ndim == 0:
            masses = np.full((self._n_particles,), float(masses))
        elif masses.ndim == 1 and masses.shape[0] != self._n_particles:
            raise ValueError("Length of m must equal particles_cnt")
        self._m = masses

        # Initialize positions uniformly in unit square, avoiding walls
        self._r = np.random.uniform(low=0.0 + R, high=1.0 - R, size=(2, self._n_particles))

        # Draw initial velocities from Maxwell distribution at temperature T
        # Use the scaled k_boltz constant for consistency with original code
        sigma = np.sqrt(self._k_boltz * T / self._m)
        self._v = stats.norm.rvs(loc=0.0, scale=sigma, size=(2, self._n_particles))

        # Save initial target temperature and energy
        self._potential_energy = []
        self._kinetic_energy = []
        self._E_full: float = self.calc_full_energy()
        self._T_tar: float = self.T

        # Thermal wall parameters
        self.T_left: float = thermal_cfg.T_left
        self.T_right: float = thermal_cfg.T_right
        self.accommodation: float = thermal_cfg.accommodation

        # Prepare collision pairs for particle collisions
        self._init_ids_pairs()

        # Frame counter for energy fixes (unused, kept for API compatibility)
        self._frame_no: int = 1
        # Base integration time step and current scaling factor (slow-mo support)
        self._base_dt: float = TIME_STEP
        self._time_scale: float = 1.0
        self._dt: float = self._base_dt * self._time_scale

        # Track particle indices that touched each wall during the last step
        self._last_wall_hits: Dict[str, np.ndarray] = {
            'left': np.empty(0, dtype=int),
            'right': np.empty(0, dtype=int),
            'top': np.empty(0, dtype=int),
            'bottom': np.empty(0, dtype=int),
        }
        self._midplane_position: float = 0.5
        self._last_midplane_flux: float = 0.0
        self._last_midplane_crossings: Dict[str, np.ndarray] = {
            'left_to_right': np.empty(0, dtype=int),
            'right_to_left': np.empty(0, dtype=int),
        }
        self._last_midplane_counts: Dict[str, int] = {
            'left_to_right': 0,
            'right_to_left': 0,
        }
        self._flux_history: list[Tuple[float, float]] = []
        self._max_flux_history: int = 5000
        self._elapsed_time: float = 0.0

    # -------------------------------------------------------------------------
    # Properties to expose slices of the state
    @property
    def r(self) -> ndarray:
        """Return positions of gas particles as a 2×N array."""
        return self._r

    @property
    def r_spring(self) -> ndarray:
        """Return positions of spring particles (empty array)."""
        return np.zeros((2, 0), dtype=float)

    @property
    def v(self) -> ndarray:
        """Return velocities of gas particles as a 2×N array."""
        return self._v

    @property
    def v_spring(self) -> ndarray:
        """Return velocities of spring particles (empty array)."""
        return np.zeros((2, 0), dtype=float)

    @property
    def m(self) -> ndarray:
        """Return masses of gas particles."""
        return self._m

    @property
    def m_spring(self) -> ndarray:
        """Return masses of spring particles (empty array)."""
        return np.zeros((0,), dtype=float)

    @property
    def R(self) -> float:
        """Radius of gas particles."""
        return self._R

    @property
    def R_spring(self) -> float:
        """Radius of spring particles (zero since none exist)."""
        return 0.0

    # -------------------------------------------------------------------------
    # Time scaling helpers ----------------------------------------------------
    def set_time_scale(self, scale: float) -> None:
        """Adjust integration step multiplier used for slow-motion or speed-up."""
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        if not math.isfinite(scale):
            scale = 1.0
        # Clamp to a sensible range to keep the integrator stable
        scale = float(np.clip(scale, 0.05, 5.0))
        self._time_scale = scale
        self._dt = self._base_dt * self._time_scale

    def get_time_scale(self) -> float:
        """Return the current integration step multiplier."""
        return self._time_scale

    def get_last_wall_hits(self) -> Dict[str, np.ndarray]:
        """Return indices of particles that touched each wall during the last step."""
        return {side: hits.copy() for side, hits in self._last_wall_hits.items()}

    def get_last_midplane_flux(self) -> float:
        """Return the most recent net flux through the midplane."""
        return self._last_midplane_flux

    def get_last_midplane_counts(self) -> Dict[str, int]:
        """Return counts of crossings through the midplane for the last step."""
        return self._last_midplane_counts.copy()

    def get_midplane_crossings(self) -> Dict[str, np.ndarray]:
        """Return indices of particles that crossed the midplane in the last step."""
        return {side: indices.copy() for side, indices in self._last_midplane_crossings.items()}

    def get_midplane_flux_history(self) -> list[Tuple[float, float]]:
        """Return a shallow copy of the recorded flux history."""
        return list(self._flux_history)

    def get_elapsed_time(self) -> float:
        """Return the total elapsed simulation time."""
        return self._elapsed_time

    # -------------------------------------------------------------------------
    # Thermodynamic properties
    @property
    def T(self) -> float:
        """Compute instantaneous temperature from kinetic energies (K)."""
        # The factor 2 accounts for 2 degrees of freedom per particle
        return np.mean((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / (2 * self._k_boltz)

    @T.setter
    def T(self, val: float) -> None:
        if val <= 0:
            raise ValueError("Temperature must be positive")
        delta = val / self._T_tar
        # Scale velocities to achieve new temperature
        self._v *= np.sqrt(delta)
        self._E_full = self.calc_full_energy()
        self._T_tar = val

    # -------------------------------------------------------------------------
    def _init_ids_pairs(self) -> None:
        """Compute index pairs for potential collisions between gas particles."""
        particles_ids = np.arange(self._n_particles)
        self._particles_ids_pairs = np.asarray(list(itertools.combinations(particles_ids, 2)), dtype=int)
        self._available_particles_ids_pairs = self._particles_ids_pairs.copy()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_deltad2_pairs(r: np.ndarray, ids_pairs: np.ndarray) -> np.ndarray:
        """Compute squared distances between all pairs of points given by indices."""
        dx = np.diff(np.stack([r[0][ids_pairs[:, 0]], r[0][ids_pairs[:, 1]]]).T).squeeze()
        dy = np.diff(np.stack([r[1][ids_pairs[:, 0]], r[1][ids_pairs[:, 1]]]).T).squeeze()
        return dx ** 2 + dy ** 2

    @staticmethod
    def compute_new_v(
        v1: np.ndarray, v2: np.ndarray, r1: np.ndarray, r2: np.ndarray, m1: np.ndarray, m2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute post‑collision velocities for an elastic collision of two particles."""
        m_s = m1 + m2
        dr = r1 - r2
        dr_norm_sq = np.linalg.norm(dr, axis=0) ** 2
        dot1 = np.sum((2 * m2 / m_s) * (v1 - v2) * dr, axis=0)
        dot2 = np.sum((2 * m1 / m_s) * (v2 - v1) * dr, axis=0)
        v1new = v1 - (dot1 * dr) / dr_norm_sq
        v2new = v2 - (dot2 * dr) / dr_norm_sq
        return v1new, v2new

    # -------------------------------------------------------------------------
    def motion(self, dt: float) -> float:
        """Advance the system by one time step of length ``dt``.

        Returns
        -------
        float
            Always returns ``0.0`` in this version.  In the original code
            this value was the work done by the spring force.
        """
        # ------------------------------------------------------------------
        # Handle particle–particle collisions
        # Only consider pairs that haven't recently collided
        if self._available_particles_ids_pairs.size:
            # Determine which pairs are colliding based on overlap
            d2 = self.get_deltad2_pairs(self._r, self._available_particles_ids_pairs)
            # A collision occurs when the squared distance is less than the sum of radii squared
            colliding_mask = d2 < (2 * self._R) ** 2
            ic_particles = self._available_particles_ids_pairs[colliding_mask]
        else:
            ic_particles = np.zeros((0, 2), dtype=int)

        # Update the available pairs: once a pair has collided it is removed
        if ic_particles.size:
            # Remove collided pairs from the available list
            self._available_particles_ids_pairs = self._update_available(
                self._particles_ids_pairs, ic_particles
            )
            # Resolve collisions by updating velocities
            v1 = self._v[:, ic_particles[:, 0]]
            v2 = self._v[:, ic_particles[:, 1]]
            r1 = self._r[:, ic_particles[:, 0]]
            r2 = self._r[:, ic_particles[:, 1]]
            m1 = self._m[ic_particles[:, 0]]
            m2 = self._m[ic_particles[:, 1]]
            v1new, v2new = self.compute_new_v(v1, v2, r1, r2, m1, m2)
            self._v[:, ic_particles[:, 0]] = v1new
            self._v[:, ic_particles[:, 1]] = v2new

            # ------------------------------------------------------------------
            # Positional correction: separate overlapping particles
            # After updating velocities, particles may still overlap because they
            # penetrated each other within one time step.  To prevent "sticking"
            # and repeated collisions, move them apart along the line of centers.
            idx_i = ic_particles[:, 0]
            idx_j = ic_particles[:, 1]
            # Vector from j to i for each colliding pair
            dr = self._r[:, idx_i] - self._r[:, idx_j]  # shape (2, K)
            # Euclidean distance between centers
            dist = np.linalg.norm(dr, axis=0)
            # Compute how much they overlap: (2R - dist).  Negative values mean no overlap
            overlap = (2.0 * self._R) - dist
            # Mask of truly overlapping pairs (distance < 2R)
            overlap_mask = overlap > 0.0
            if np.any(overlap_mask):
                # Normalize the direction vector for overlapping pairs
                # To avoid division by zero, clip very small distances
                safe_dist = np.copy(dist[overlap_mask])
                safe_dist[safe_dist < 1e-12] = 1e-12
                n = dr[:, overlap_mask] / safe_dist  # shape (2, M)
                # Each particle moves half the overlap distance in opposite directions
                shift = 0.5 * overlap[overlap_mask]
                # Broadcast shift to both components
                self._r[:, idx_i[overlap_mask]] += n * shift
                self._r[:, idx_j[overlap_mask]] -= n * shift

        # ------------------------------------------------------------------
        # Handle wall collisions: determine which particles hit which wall
        x = self._r[0]
        y = self._r[1]
        vx = self._v[0]
        vy = self._v[1]
        masses = self._m

        min_x = self._R
        max_x = 1.0 - self._R
        min_y = self._R
        max_y = 1.0 - self._R

        # Compute masks for each wall.  Particles must be moving toward the wall.
        mask_left = (x <= min_x) & (vx < 0.0)
        mask_right = (x >= max_x) & (vx > 0.0)
        mask_bottom = (y <= min_y) & (vy < 0.0)
        mask_top = (y >= max_y) & (vy > 0.0)

        # Remember which particles touched the walls this step
        self._last_wall_hits['left'] = np.where(mask_left)[0]
        self._last_wall_hits['right'] = np.where(mask_right)[0]
        self._last_wall_hits['bottom'] = np.where(mask_bottom)[0]
        self._last_wall_hits['top'] = np.where(mask_top)[0]

        # Reposition any particles that penetrated the wall back onto the boundary
        if np.any(mask_left):
            x[mask_left] = min_x
        if np.any(mask_right):
            x[mask_right] = max_x
        if np.any(mask_bottom):
            y[mask_bottom] = min_y
        if np.any(mask_top):
            y[mask_top] = max_y

        # Apply reflection or thermalization
        reflect_with_accommodation(vx, vy, 'left', self.T_left, self.accommodation, masses, mask_left)
        reflect_with_accommodation(vx, vy, 'right', self.T_right, self.accommodation, masses, mask_right)
        reflect_with_accommodation(vx, vy, 'bottom', None, 0.0, masses, mask_bottom)
        reflect_with_accommodation(vx, vy, 'top', None, 0.0, masses, mask_top)

        # ------------------------------------------------------------------
        # Integrate positions
        prev_x = self._r[0].copy()
        self._r += self._v * dt
        self._update_midplane_flux(prev_x, dt)

        # ------------------------------------------------------------------
        return 0.0

    # -------------------------------------------------------------------------
    def __iter__(self) -> 'Simulation':
        return self

    def __next__(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, float]:
        """Advance the simulation and return state arrays.

        The tuple contains positions and velocities of gas particles and
        (empty) spring particles, plus the value returned by ``motion``.
        """
        f = self.motion(dt=self._dt)
        self._frame_no = (self._frame_no + 1) % 5

        self._potential_energy.append(self.calc_potential_energy())
        self._kinetic_energy.append(self.calc_kinetic_energy())

        if self._frame_no == 0:
            self._fix_energy()

        return self.r, self.r_spring, self.v, self.v_spring, f

    # -------------------------------------------------------------------------
    def _update_midplane_flux(self, prev_x: np.ndarray, dt: float) -> None:
        """Update net particle flux across the midplane based on the last step."""
        plane = self._midplane_position
        new_x = self._r[0]
        prev_x_arr = np.asarray(prev_x, dtype=float)
        if prev_x_arr.shape != new_x.shape:
            prev_x_arr = np.reshape(prev_x_arr, new_x.shape)
        left_to_right_mask = (prev_x_arr < plane) & (new_x >= plane)
        right_to_left_mask = (prev_x_arr > plane) & (new_x <= plane)
        left_to_right = np.nonzero(left_to_right_mask)[0]
        right_to_left = np.nonzero(right_to_left_mask)[0]

        self._last_midplane_crossings['left_to_right'] = left_to_right
        self._last_midplane_crossings['right_to_left'] = right_to_left

        ltr_count = int(left_to_right.size)
        rtl_count = int(right_to_left.size)
        self._last_midplane_counts['left_to_right'] = ltr_count
        self._last_midplane_counts['right_to_left'] = rtl_count

        net_crossings = ltr_count - rtl_count
        self._last_midplane_flux = net_crossings / dt if dt > 0.0 else 0.0

        # Record time and flux samples for later analysis/visualisation
        if dt > 0.0:
            self._elapsed_time += dt
        timestamp = self._elapsed_time
        self._flux_history.append((timestamp, self._last_midplane_flux))
        if len(self._flux_history) > self._max_flux_history:
            self._flux_history.pop(0)

    # -------------------------------------------------------------------------
    def _update_available(self, arr: ndarray, sub_arr: ndarray) -> ndarray:
        """Remove rows of ``arr`` that appear in ``sub_arr`` (order insensitive)."""
        if arr.size == 0:
            return arr
        eq_mat = sub_arr.T[None, ...] == arr[..., None]
        na_idx = np.any(np.all(eq_mat, axis=1), axis=1)
        return arr[~na_idx, :]

    # -------------------------------------------------------------------------
    def add_particles(self, r: ndarray, v: ndarray, m: ndarray) -> None:
        """Add new gas particles to the simulation.

        Parameters
        ----------
        r: ndarray
            Positions of the new particles, shape (2, N_new).
        v: ndarray
            Velocities of the new particles, shape (2, N_new).
        m: ndarray
            Masses of the new particles, shape (N_new,).
        """
        if r.shape != v.shape or r.shape[0] != 2 or r.shape[1] != m.shape[0]:
            raise ValueError("Shapes of r, v and m are inconsistent")
        self._r = np.hstack([self._r, r])
        self._v = np.hstack([self._v, v])
        self._m = np.hstack([self._m, m])
        self._n_particles += r.shape[1]
        # Recompute collision pairs
        self._init_ids_pairs()
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    # -------------------------------------------------------------------------
    def _set_particles_cnt(self, particles_cnt: int) -> None:
        """Reset the number of gas particles to the given count.

        If the new count is smaller than the current count, particles are
        removed from the end.  If it is larger, new particles are added
        with random positions and velocities sampled from the current
        speed distribution and median mass.
        """
        if particles_cnt < 0:
            raise ValueError("particles_cnt must be >= 0")
        if particles_cnt < self._n_particles:
            idx = slice(particles_cnt)
            self._r = self._r[:, idx]
            self._v = self._v[:, idx]
            self._m = self._m[idx]
        if particles_cnt > self._n_particles:
            new_cnt = particles_cnt - self._n_particles
            # Positions uniformly distributed away from walls
            new_r = np.random.uniform(low=0.0 + self._R, high=1.0 - self._R, size=(2, new_cnt))
            # Velocities sampled from current distribution
            if self._n_particles > 0:
                v_std = np.std(self._v, axis=1)
            else:
                v_std = np.array([1.0, 1.0])
            new_v = np.stack([
                rng.normal(0.0, v_std[0], size=new_cnt) if rng is not None else np.zeros(new_cnt),
                rng.normal(0.0, v_std[1], size=new_cnt) if rng is not None else np.zeros(new_cnt),
            ])
            new_m = np.full((new_cnt,), np.median(self._m) if self._m.size > 0 else 1.0)
            self.add_particles(new_r, new_v, new_m)
        if particles_cnt != self._n_particles:
            self._n_particles = particles_cnt
            self._init_ids_pairs()
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    # -------------------------------------------------------------------------
    def set_params(
        self,
        gamma: float = None,
        k: float = None,
        l_0: float = None,
        R: float = None,
        T: float = None,
        m: float = None,
        particles_cnt: int = None,
        T_left: float = None,
        T_right: float = None,
        accommodation: float = None,
    ) -> None:
        """Update simulation parameters on the fly.

        Parameters correspond to those accepted by the constructor.  Any
        parameter passed as ``None`` will be left unchanged.
        """
        if gamma is not None:
            self._gamma = float(gamma)
        if k is not None:
            self._k = float(k)
        if l_0 is not None:
            self._l_0 = float(l_0)
        if R is not None:
            self._R = float(R)
        if T is not None:
            self.T = float(T)
        if m is not None:
            if m <= 0:
                raise ValueError("m must be > 0")
            self._m[:] = float(m)
        if particles_cnt is not None:
            self._set_particles_cnt(int(particles_cnt))
        if T_left is not None:
            self.T_left = float(T_left)
        if T_right is not None:
            self.T_right = float(T_right)
        if accommodation is not None:
            self.accommodation = float(np.clip(accommodation, 0.0, 1.0))
        # Recompute full energy after parameter changes
        self._E_full = self.calc_full_energy()
        self._T_tar = self.T

    # -------------------------------------------------------------------------
    def expected_potential_energy(self) -> float:
        """Return zero since no external potential exists."""
        return 0.0

    def expected_kinetic_energy(self) -> float:
        """Return the expected kinetic energy per particle (k_B T)."""
        return float(self._k_boltz * self._T_tar)

    def calc_kinetic_energy(self) -> float:
        """Calculate mean kinetic energy of gas particles."""
        return np.mean((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / 2.0

    def calc_full_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of gas particles."""
        return np.sum((np.linalg.norm(self._v, axis=0) ** 2) * self._m) / 2.0

    def _fix_energy(self) -> None:
        """Gently counteract numerical drift without blocking heat exchange.

        The total energy should change only due to wall interactions or
        deliberate parameter updates.  We therefore track a slowly varying
        target energy and only rescale velocities when the instantaneous
        energy deviates slightly from that target (typical of integration
        error).  Substantial changes driven by the walls are preserved.
        """
        current_E = self.calc_full_energy()
        if current_E <= 0.0:
            return

        if self._E_full <= 0.0:
            self._E_full = current_E
            return

        relax = 0.1
        self._E_full = (1.0 - relax) * self._E_full + relax * current_E
        scale = self._E_full / current_E
        if scale <= 0.0:
            return
        scale = math.sqrt(scale)
        if abs(scale - 1.0) < 0.05:
            self._v *= scale

    def calc_full_energy(self) -> float:
        """Return the total energy (purely kinetic)."""
        return self.calc_full_kinetic_energy()

    def calc_potential_energy(self) -> float:
        """Return zero since there is no potential energy."""
        return 0.0

    def mean_potential_energy(self, frames_c: Union[int, None] = None) -> float:
        """Always return zero in absence of potential energy."""
        return 0.0

    def mean_kinetic_energy(self, frames_c: Union[int, None] = None) -> float:
        """Return the mean of the stored kinetic energy history."""
        if frames_c is None:
            if not self._kinetic_energy:
                return 0.0
            return float(np.mean(self._kinetic_energy))
        else:
            if not self._kinetic_energy:
                return 0.0
            return float(np.mean(self._kinetic_energy[-frames_c:]))

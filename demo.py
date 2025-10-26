"""
Modified Demo module without spring particles.

This version of ``demo.py`` interfaces with the updated ``Simulation``
class that lacks spring particles.  It removes all logic associated with
the spring particle pair and renders only the gas particles.  The
parameters for the spring (``R``, ``m_spring``) are ignored.  The
potential energy arrays remain for compatibility but will contain zeroes.
"""

import pygame
import numpy as np
import config
from simulation import Simulation


class Demo:
    def __init__(self, app, position, demo_size, bg_color, border_color, bg_screen_color, params):
        """
        Initialize a new demonstration instance.

        Parameters
        ----------
        app : App
            Reference to the parent application containing the pygame screen.
        position : tuple
            (x, y) coordinates of the top‑left corner of the simulation box.
        demo_size : tuple
            (width, height) of the simulation area in pixels.
        bg_color : tuple
            RGB background colour of the simulation area.
        border_color : tuple
            RGB colour of the border around the simulation area.
        bg_screen_color : tuple
            Colour used for the masked outer border region.
        params : dict
            Dictionary of initial simulation parameters (gamma, k, R, T, r, etc.).
        """
        self.screen = app.screen
        self.bg_color = bg_color
        self.bg_screen_color = bg_screen_color
        self.bd_color = border_color
        self.position = position
        # Pygame rect describing the simulation area
        self.main = pygame.Rect(*position, *demo_size)
        # Store individual dimensions so the demo can be rectangular
        self.width, self.height = demo_size
        # Pixel coordinates of the bottom‑left corner of the simulation area (used for transforms)
        self.pos_start = position[0], position[1] + self.height
        # Copy of the initial parameter values used by sliders
        # Copy of the initial parameter values used by sliders.  Remove keys
        # corresponding to unused simulation parameters (gamma, k, mass of spring,
        # spring radius, etc.) to avoid storing unneeded data.  If those keys
        # are not present, ``pop`` simply returns ``None``.  This keeps
        # ``self.params`` compact and eliminates references to unused legacy
        # spring parameters.
        self.params = dict(params)
        for unused_key in ('gamma', 'k', 'm_spring', 'R', 'R_spring', 'radius_scale'):
            self.params.pop(unused_key, None)
        self.modified_par = None
        loader = config.ConfigLoader()

        # Factor by which the physical radius is scaled up.  Multiplying
        # ``loader["R_size"]`` by this factor increases the actual
        # collision radius of the particles.  Set the default scale to
        # 1.0 so that particles start with the nominal radius from
        # configuration.  Increasing the physical radius while keeping
        # time step constant may require care in simulation stability;
        # positional correction is already in place.
        self.physical_radius_scale = 1.0

        # Factor to scale the physical radius when drawing.  Setting this
        # near 1.0 makes the particles appear the same size as their
        # physical collision radius.  Lower values shrink them visually.
        self.draw_radius_factor = 1.0

        # Keep track of indices of tagged (marked) particles.  These will
        # be drawn in a distinct colour and can be analysed separately.
        self.tagged_indices: list[int] = []
        # Colour used for tagged particles.  Tagged particles will be
        # drawn in a distinct colour.  According to the latest
        # requirements the marked particles should appear bright yellow
        # to stand out against the temperature gradient colours.
        self.tagged_color: tuple[int, int, int] = (255, 165, 0)

        # Masses for gas particles only.  ``self.params['r']`` specifies the
        # number of gas particles; there are no spring masses in this
        # simplified model.  Masses are drawn from configuration.
        m = np.ones((self.params['r'],), dtype=float) * loader["R_mass"]

        # Legacy parameters (gamma, k, l_0) are passed through for API compatibility
        l_0 = loader['l_0']
        # Cache the base particle radius from the configuration.  ``R_size``
        # defines the nominal physical radius (in box units).  By
        # storing this value we can compute new radii when the user
        # adjusts the particle size via the UI.
        self.base_radius = loader["R_size"]
        # Initialize the simulation.  Multiply the base radius by our
        # physical scale factor to enlarge the physical collision size.
        # Legacy parameters may be absent in ``params`` because the UI
        # does not expose sliders for them.  Fetch them with default
        # fallbacks.  ``gamma`` and ``k`` are unused in the current
        # simulation but accepted for API compatibility.
        gamma_val = params.get('gamma', 1.0)
        k_val = params.get('k', 1.0)
        # Create the simulation with legacy parameters, particle radius and counts.
        self.simulation = Simulation(
            gamma=gamma_val,
            k=k_val,
            l_0=l_0,
            R=self.base_radius * self.physical_radius_scale,
            particles_cnt=self.params['r'],
            T=self.params['T'],
            m=m,
        )

        # If thermal wall parameters are provided in params, set them on the simulation
        # (These keys may not exist in older configs.)
        t_left = params.get('T_left')
        t_right = params.get('T_right')
        accom = params.get('accommodation')
        if t_left is not None or t_right is not None or accom is not None:
            self.simulation.set_params(
                T_left=t_left,
                T_right=t_right,
                accommodation=accom,
            )

    def update_radius_scale(self, scale: float) -> None:
        """
        Update both the physical and visual radius scales of the particles.

        This method is intended to be called from the demo screen when
        the user changes the particle size slider.  It sets the
        ``physical_radius_scale`` and ``draw_radius_factor`` to the same
        value, updates the simulation's physical radius accordingly,
        and stores the new scale.  Changing the physical radius
        influences collision detection, while changing the draw factor
        adjusts the rendered size on screen.  Both are applied at once
        so that the user perceives the change consistently.

        Parameters
        ----------
        scale : float
            New scaling factor (1.0 means default size; higher values
            enlarge particles; lower values shrink them).
        """
        # Avoid zero or negative scales to keep a valid radius.
        if scale < 0.1:
            scale = 0.1
        # Store the new scale for both physical interactions and drawing
        self.physical_radius_scale = scale
        self.draw_radius_factor = scale
        # Compute the new physical radius using the cached base radius
        new_R = self.base_radius * self.physical_radius_scale
        # Update the simulation with the new radius
        self.simulation.set_params(R=new_R)

    def set_params(self, params, par):
        # Dispatch updated simulation parameters based on the changed
        # parameter name.  Legacy parameters such as ``gamma`` and ``k``
        # are no longer processed, because they have no effect in the
        # current model.  Updates to particle size are handled in the
        # DemoScreen via ``update_radius_scale``.
        if par == 'T':
            self.simulation.set_params(T=params['T'])
        elif par == 'r':
            self.simulation.set_params(particles_cnt=params['r'])
        # ignore any other parameters (gamma, k, R, etc.)

    def draw_check(self, params):
        # Draw background box
        pygame.draw.rect(self.screen, self.bg_color, self.main)
        # Detect parameter changes
        for i, par1, par2 in zip(range(len(self.params)), params['params'].values(), self.params.values()):
            if abs(par1 - par2) > 1e-4:
                self.modified_par = list(self.params.keys())[i]
                params['is_changed'] = True
                break

        # Advance simulation and record energies
        loader = config.ConfigLoader()
        new_args = next(self.simulation)
        for i in range(params['params']['speed']):
            new_args = next(self.simulation)
            # Assign direct float values; the simulation returns Python floats
            params['kinetic'][i] = self.simulation.calc_kinetic_energy()
            params['potential'][i] = self.simulation.calc_potential_energy()
            params['mean_kinetic'][i] = self.simulation.mean_kinetic_energy(loader['sim_avg_frames_c'])
            params['mean_potential'][i] = self.simulation.mean_potential_energy(loader['sim_avg_frames_c'])
        for i in range(params['params']['speed'], len(params['kinetic'])):
            params['kinetic'][i] = -1
            params['potential'][i] = -1
            params['mean_kinetic'][i] = -1
            params['mean_potential'][i] = -1

        # Unpack positions; r_spring is empty
        r = new_args[0].copy()
        # Determine draw radii
        # Convert physical radius (0–1) to pixel radius.  Multiply by
        # ``draw_radius_factor`` to make particles visually smaller than
        # their physical size to reduce apparent overlap in the UI.
        scale = min(self.width, self.height)
        r_radius = scale * self.simulation.R * self.draw_radius_factor
        # Transform positions from unit box to screen coordinates
        r[0] = self.pos_start[0] + r[0] * self.width
        r[1] = self.pos_start[1] - r[1] * self.height
        r = np.round(r)
        r_radius = max(1, int(round(r_radius)))
        # Compute velocities and speed magnitudes for color mapping
        # new_args[2] corresponds to velocities (2×N array)
        v = new_args[2] if isinstance(new_args, (list, tuple)) and len(new_args) >= 3 else self.simulation.v
        # Speed (scalar) per particle
        speeds = np.linalg.norm(v, axis=0)
        # Normalize speeds between 0 and 1 for color interpolation
        if speeds.size > 0:
            v_min = float(np.min(speeds))
            v_max = float(np.max(speeds))
            # Avoid division by zero
            denom = v_max - v_min if v_max > v_min else 1.0
            normalized = (speeds - v_min) / denom
        else:
            normalized = np.zeros_like(speeds)
        # Precompute a set for faster membership checks of tagged particles
        tagged_set = set(self.tagged_indices)
        # Draw gas particles: use separate colour for tagged particles
        for idx in range(r.shape[1]):
            if idx in tagged_set:
                # Tagged particles are drawn in a distinct colour (self.tagged_color)
                color = self.tagged_color
            else:
                # Map speed to colour: blue (cold) → red (hot)
                c = float(normalized[idx])
                red   = int(255 * c)
                green = 0
                blue  = int(255 * (1.0 - c))
                color = (red, green, blue)
            pygame.draw.circle(self.screen, color, tuple(r[:, idx]), r_radius)
        # Draw border
        inner_border = 3
        mask_border = 50
        pygame.draw.rect(
            self.screen,
            self.bg_screen_color,
            (
                self.position[0] - mask_border,
                self.position[1] - mask_border,
                self.width + mask_border * 2,
                self.height + mask_border * 2,
            ),
            mask_border,
        )
        pygame.draw.rect(
            self.screen,
            self.bd_color,
            (
                self.position[0] - inner_border,
                self.position[1] - inner_border,
                self.width + inner_border * 2,
                self.height + inner_border * 2,
            ),
            inner_border,
        )

    def _refresh_iter(self, params):
        if self.modified_par is not None:
            self.set_params(params['params'], self.modified_par)
            self.params[self.modified_par] = params['params'][self.modified_par]

    # -----------------------------------------------------------------
    def add_tagged_particles(self, count: int) -> None:
        """
        Add a specified number of tagged (coloured) particles at the
        centre of the box.

        The new particles are placed at the centre of the simulation
        domain (``x = 0.5``, ``y = 0.5``) with random jitter to avoid
        immediate overlap.  Their velocities are sampled from the
        current gas temperature distribution.  The indices of these
        particles are recorded in ``self.tagged_indices`` so that they
        can be drawn in a different colour.

        Parameters
        ----------
        count : int
            Number of particles to add.
        """
        # Determine the starting index for new particles
        n_old = self.simulation._n_particles
        # Build positions at the box centre with small random jitter
        jitter = 0.001  # small displacement to avoid stacking
        # Uniformly distribute jitter within a tiny square around centre
        r_new = np.tile(np.array([[0.5], [0.5]]), (1, count))
        if count > 0:
            r_new = r_new + (np.random.uniform(low=-jitter, high=jitter, size=(2, count)))
        # Ensure the new particles respect the walls (stay within [R, 1-R])
        # Clip in case jitter pushes them outside
        R_phys = self.simulation.R
        r_new[0] = np.clip(r_new[0], R_phys, 1.0 - R_phys)
        r_new[1] = np.clip(r_new[1], R_phys, 1.0 - R_phys)
        # Sample velocities consistent with current temperature
        # Compute standard deviation for Maxwell distribution using scaled k_boltz
        # Use median mass in case of varying masses
        masses = self.simulation.m
        if masses.size > 0:
            m_typ = np.median(masses)
        else:
            m_typ = 1.0
        sigma = np.sqrt(self.simulation._k_boltz * self.simulation.T / m_typ)
        v_new = np.random.normal(loc=0.0, scale=sigma, size=(2, count))
        # Masses for new particles
        m_new = np.full((count,), m_typ)
        # Add to simulation
        self.simulation.add_particles(r_new, v_new, m_new)
        # Record tagged indices
        new_indices = list(range(n_old, n_old + count))
        self.tagged_indices.extend(new_indices)
        # Update parameter dictionary to reflect increased number of particles
        if 'r' in self.params:
            self.params['r'] += count

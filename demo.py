"""
Modified Demo module without spring particles.

This version of ``demo.py`` interfaces with the updated ``Simulation``
class that lacks spring particles.  It removes all logic associated with
the spring particle pair and renders only the gas particles.  The
parameters for the spring (``R``, ``m_spring``) are ignored.  The
potential energy arrays remain for compatibility but will contain zeroes.
"""

import math
import pygame
import numpy as np
import config
from typing import Optional
from simulation import Simulation


class Demo:
    def __init__(
        self,
        app,
        position,
        demo_size,
        bg_color,
        border_color,
        bg_screen_color,
        params,
        wall_temp_bounds: tuple[float, float] | None = None,
    ):
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
        if wall_temp_bounds and wall_temp_bounds[1] > wall_temp_bounds[0]:
            self.wall_temp_bounds = (float(wall_temp_bounds[0]), float(wall_temp_bounds[1]))
        else:
            self.wall_temp_bounds = (100.0, 1000.0)
        self._wall_colors = {'left': (220, 70, 40), 'right': (60, 130, 255)}
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

        # Scale factors for physical collisions and rendering.  Start 30% larger
        # to make particles more visible by default.
        initial_scale = float(params.get('size_scale', 1.3))
        if initial_scale < 0.1:
            initial_scale = 0.1
        self.physical_radius_scale = initial_scale
        self.draw_radius_factor = initial_scale

        # Keep track of indices of tagged (marked) particles.  These will
        # be drawn in a distinct colour and can be analysed separately.
        self.tagged_indices: list[int] = []
        # Colour used for tagged particles.  Tagged particles will be
        # drawn in a distinct colour.  According to the latest
        # requirements the marked particles should appear bright yellow
        # to stand out against the temperature gradient colours.
        self.tagged_color: tuple[int, int, int] = (255, 220, 0)
        # Rendering helpers for highlight/dim features
        self.dim_untracked: bool = False
        self.dim_color: tuple[int, int, int] = (90, 90, 100)

        # Trajectory tracking for a tagged particle
        self.trail_enabled: bool = False
        self.trail_points: list[tuple[int, int]] = []
        self.max_trail_points: int = 600
        self.tracked_particle_id: Optional[int] = None

        # Counters for wall contacts by tagged particles
        self.wall_hits = {'left': 0, 'right': 0}
        # Flux tracking across the midplane (x = 0.5)
        self.midplane_position: float = 0.5
        self.midplane_flux_samples: list[tuple[float, float]] = []
        self.max_flux_samples: int = 2000
        self.reset_wall_hit_counters()

        # Slow-motion control: multiplier applied to the integrator step
        self.time_scale: float = float(params.get('slowmo', 1.0))

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

        # Apply the initial slow-motion factor after the simulation is created.
        self.simulation.set_time_scale(self.time_scale)

    def _temperature_to_color(self, temperature: float) -> tuple[int, int, int]:
        """
        Map a wall temperature to the same blue-to-red gradient used for particles.
        """
        min_temp, max_temp = self.wall_temp_bounds
        span = max(1e-6, max_temp - min_temp)
        norm = (float(temperature) - min_temp) / span
        norm = max(0.0, min(1.0, norm))
        red = int(round(255 * norm))
        blue = int(round(255 * (1.0 - norm)))
        return red, 0, blue

    def get_wall_color(self, side: str) -> tuple[int, int, int]:
        """Return the most recently rendered colour for the given wall."""
        return self._wall_colors.get(side, (255, 255, 255))

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

    def set_time_scale(self, scale: float) -> None:
        """Expose slow-motion control for the simulation loop."""
        if scale < 0.05:
            scale = 0.05
        self.time_scale = float(scale)
        self.simulation.set_time_scale(self.time_scale)

    def resize_viewport(self, position: tuple[int, int], demo_size: tuple[int, int]) -> None:
        """Adjust the rendering viewport to a new rectangle."""
        self.position = position
        self.main = pygame.Rect(*position, *demo_size)
        self.width, self.height = demo_size
        self.pos_start = position[0], position[1] + self.height
        self.screen = pygame.display.get_surface()

    def set_dim_untracked(self, dim: bool) -> None:
        """Enable or disable dimming of untagged particles."""
        self.dim_untracked = bool(dim)

    def set_trail_enabled(self, enabled: bool) -> None:
        """Toggle trajectory rendering for the tracked tagged particle."""
        new_state = bool(enabled)
        if new_state and not self.trail_enabled:
            # Reset trail when enabling to avoid stale points.
            self.trail_points.clear()
        if not new_state:
            self.trail_points.clear()
        self.trail_enabled = new_state
        self._ensure_tracked_particle()

    def has_tagged_particles(self) -> bool:
        """Return True when at least one tagged particle exists."""
        return bool(self.tagged_indices)

    def get_wall_hit_counts(self) -> tuple[int, int]:
        """Return accumulated counts of tagged particles hitting left/right walls."""
        return self.wall_hits['left'], self.wall_hits['right']

    def reset_wall_hit_counters(self) -> None:
        """Clear stored hit counts for both walls."""
        self.wall_hits['left'] = 0
        self.wall_hits['right'] = 0

    def _ensure_tracked_particle(self) -> None:
        """Ensure the tracked particle id refers to an existing tagged particle."""
        if not self.tagged_indices:
            self.tracked_particle_id = None
            return
        if self.tracked_particle_id not in self.tagged_indices:
            # Prefer the earliest tagged particle for consistency.
            self.tracked_particle_id = self.tagged_indices[0]
            self.trail_points.clear()

    def _update_wall_hit_counters(self) -> None:
        """Increment counters when tagged particles touch left/right walls."""
        if not self.tagged_indices:
            return
        tagged_set = set(self.tagged_indices)
        hits = self.simulation.get_last_wall_hits()
        for idx in hits.get('left', []):
            if int(idx) in tagged_set:
                self.wall_hits['left'] += 1
        for idx in hits.get('right', []):
            if int(idx) in tagged_set:
                self.wall_hits['right'] += 1

    def _record_trail_point(self, pixel_positions: np.ndarray) -> None:
        """Append the current screen-space position of the tracked particle."""
        if not self.trail_enabled:
            return
        self._ensure_tracked_particle()
        if self.tracked_particle_id is None:
            return
        idx = int(self.tracked_particle_id)
        if idx >= pixel_positions.shape[1]:
            return
        point = (int(pixel_positions[0, idx]), int(pixel_positions[1, idx]))
        if self.trail_points and self.trail_points[-1] == point:
            return
        self.trail_points.append(point)
        if len(self.trail_points) > self.max_trail_points:
            self.trail_points.pop(0)

    def _store_flux_sample(self, timestamp: float, value: float) -> None:
        """Cache the latest net flux value for future visualisation."""
        if not math.isfinite(timestamp) or not math.isfinite(value):
            return
        if self.midplane_flux_samples and abs(self.midplane_flux_samples[-1][0] - timestamp) < 1e-9:
            self.midplane_flux_samples[-1] = (timestamp, value)
        else:
            self.midplane_flux_samples.append((timestamp, value))
        if len(self.midplane_flux_samples) > self.max_flux_samples:
            self.midplane_flux_samples.pop(0)

    def _draw_midplane_wall(self) -> None:
        """Render a semi-transparent dashed divider at the domain centre."""
        wall_width = max(3, int(round(self.width * 0.012)))
        dash_length = max(10, int(round(self.height * 0.06)))
        gap_length = max(6, int(round(dash_length * 0.6)))
        wall_surface = pygame.Surface((wall_width, self.height), pygame.SRCALPHA)
        center_x = wall_width // 2
        color = (182, 186, 198, 215)
        y = 0
        while y < self.height:
            end_y = min(self.height, y + dash_length)
            pygame.draw.line(wall_surface, color, (center_x, y), (center_x, end_y), wall_width)
            y = end_y + gap_length
        wall_x = self.position[0] + int(round(self.width * self.midplane_position)) - wall_width // 2
        wall_x = max(self.position[0], min(self.position[0] + self.width - wall_width, wall_x))
        self.screen.blit(wall_surface, (wall_x, self.position[1]))

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
        elif par == 'slowmo':
            self.set_time_scale(params['slowmo'])
        # ignore any other parameters (gamma, k, R, etc.)

    def draw_check(self, params):
        # Draw background box
        pygame.draw.rect(self.screen, self.bg_color, self.main)

        # Detect parameter changes and reset change flag per frame
        params['is_changed'] = False
        self.modified_par = None
        for i, par1, par2 in zip(range(len(self.params)), params['params'].values(), self.params.values()):
            if abs(par1 - par2) > 1e-4:
                self.modified_par = list(self.params.keys())[i]
                params['is_changed'] = True
                break

        # Apply slow-motion factor immediately when it changes
        slowmo_value = float(params['params'].get('slowmo', self.time_scale))
        if abs(slowmo_value - self.time_scale) > 1e-4:
            self.set_time_scale(slowmo_value)

        # Advance simulation and record energies
        loader = config.ConfigLoader()
        speed_raw = params['params'].get('speed', 1)
        try:
            steps = int(max(1, round(speed_raw)))
        except (TypeError, ValueError):
            steps = 1

        new_args = None
        for i in range(steps):
            new_args = next(self.simulation)
            self._update_wall_hit_counters()
            if i < len(params['kinetic']):
                params['kinetic'][i] = self.simulation.calc_kinetic_energy()
                params['potential'][i] = self.simulation.calc_potential_energy()
                params['mean_kinetic'][i] = self.simulation.mean_kinetic_energy(loader['sim_avg_frames_c'])
                params['mean_potential'][i] = self.simulation.mean_potential_energy(loader['sim_avg_frames_c'])
        for i in range(steps, len(params['kinetic'])):
            params['kinetic'][i] = -1
            params['potential'][i] = -1
            params['mean_kinetic'][i] = -1
            params['mean_potential'][i] = -1

        if new_args is None:
            new_args = (
                self.simulation.r,
                self.simulation.r_spring,
                self.simulation.v,
                self.simulation.v_spring,
                0.0,
            )

        # Unpack positions; r_spring is empty
        r = np.array(new_args[0], copy=True)
        # Determine draw radii. Convert physical radius (0–1) to pixel radius.
        scale = min(self.width, self.height)
        r_radius = scale * self.simulation.R * self.draw_radius_factor

        # Draw coloured wall strips before overlaying particles
        wall_thickness = max(6, int(round(self.width * 0.015)))
        left_wall_rect = pygame.Rect(self.position[0], self.position[1], wall_thickness, self.height)
        right_wall_rect = pygame.Rect(
            self.position[0] + self.width - wall_thickness,
            self.position[1],
            wall_thickness,
            self.height,
        )
        param_values = params.get('params') if isinstance(params, dict) else None
        left_temp_raw = self.simulation.T_left
        right_temp_raw = self.simulation.T_right
        if isinstance(param_values, dict):
            left_temp_raw = param_values.get('T_left', left_temp_raw)
            right_temp_raw = param_values.get('T_right', right_temp_raw)
        try:
            left_temp = float(left_temp_raw)
        except (TypeError, ValueError):
            left_temp = float(self.simulation.T_left)
        try:
            right_temp = float(right_temp_raw)
        except (TypeError, ValueError):
            right_temp = float(self.simulation.T_right)
        left_color = self._temperature_to_color(left_temp)
        right_color = self._temperature_to_color(right_temp)
        self._wall_colors['left'] = left_color
        self._wall_colors['right'] = right_color
        pygame.draw.rect(self.screen, left_color, left_wall_rect)
        pygame.draw.rect(self.screen, right_color, right_wall_rect)

        # Transform positions from unit box to screen coordinates
        r[0] = self.pos_start[0] + r[0] * self.width
        r[1] = self.pos_start[1] - r[1] * self.height
        r = np.round(r).astype(int)
        r_radius = max(1, int(round(r_radius)))

        # Compute velocities and speed magnitudes for colour mapping
        v = new_args[2] if isinstance(new_args, (list, tuple)) and len(new_args) >= 3 else self.simulation.v
        speeds = np.linalg.norm(v, axis=0)
        if speeds.size > 0:
            v_min = float(np.min(speeds))
            v_max = float(np.max(speeds))
            denom = v_max - v_min if v_max > v_min else 1.0
            normalized = (speeds - v_min) / denom
        else:
            normalized = np.zeros_like(speeds)

        # Store the current position of the tracked particle for the trail feature
        self._record_trail_point(r)
        flux_timestamp = self.simulation.get_elapsed_time()
        flux_value = self.simulation.get_last_midplane_flux()
        self._store_flux_sample(flux_timestamp, flux_value)

        # Precompute sets for tagged particles and highlight handling
        tagged_set = set(self.tagged_indices)

        # Draw gas particles with dimming/highlighting options
        for idx in range(r.shape[1]):
            point = (int(r[0, idx]), int(r[1, idx]))
            if idx in tagged_set:
                color = self.tagged_color
            elif self.dim_untracked:
                color = self.dim_color
            else:
                c = float(normalized[idx]) if normalized.size else 0.0
                red = int(255 * c)
                green = 0
                blue = int(255 * (1.0 - c))
                color = (red, green, blue)
            pygame.draw.circle(self.screen, color, point, r_radius)

        # Draw trajectory trail over the particles so it remains visible
        if self.trail_enabled and len(self.trail_points) >= 2:
            pygame.draw.lines(self.screen, self.tagged_color, False, self.trail_points, 2)

        if self.trail_enabled:
            self._ensure_tracked_particle()
        if self.tracked_particle_id is not None and self.tracked_particle_id < r.shape[1]:
            focus_point = (int(r[0, self.tracked_particle_id]), int(r[1, self.tracked_particle_id]))
            pygame.draw.circle(self.screen, (255, 255, 255), focus_point, r_radius + 3, 2)

        self._draw_midplane_wall()

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

    def draw_midplane_flux_graph(
        self,
        target_surface: pygame.Surface,
        rect: pygame.Rect,
        samples: Optional[list[tuple[float, float]]] = None,
        line_color: tuple[int, int, int] = (90, 180, 255),
        baseline_color: tuple[int, int, int] = (180, 180, 180),
        background: Optional[tuple[int, int, int, int]] = (0, 0, 0, 0),
    ) -> None:
        """
        Render a step-style graph of the recorded midplane flux values.

        The function draws onto ``target_surface`` but does not blit the
        result to the on-screen display.  This lets callers integrate the
        graph into their own layout later.
        """
        data = self.midplane_flux_samples if samples is None else samples
        rect = pygame.Rect(rect)
        if len(data) < 2 or rect.width <= 1 or rect.height <= 1:
            return

        times = [float(t) for t, _ in data]
        values = [float(v) for _, v in data]
        t_min = times[0]
        t_max = times[-1]
        if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
            return

        v_min = min(values)
        v_max = max(values)
        if not (math.isfinite(v_min) and math.isfinite(v_max)):
            return
        if v_max <= v_min:
            expand = max(1.0, abs(v_min) * 0.1)
            v_min -= expand
            v_max += expand
        else:
            span = v_max - v_min
            padding = span * 0.1
            v_min -= padding
            v_max += padding

        graph_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        if background is not None:
            graph_surface.fill(background)

        if v_min <= 0.0 <= v_max:
            zero_rel = (0.0 - v_min) / (v_max - v_min)
            zero_y = rect.height - zero_rel * rect.height
            pygame.draw.line(
                graph_surface,
                baseline_color,
                (0, int(round(zero_y))),
                (rect.width, int(round(zero_y))),
                1,
            )

        def to_point(time_value: float, flux_value: float) -> tuple[int, int]:
            x_rel = (time_value - t_min) / (t_max - t_min)
            y_rel = (flux_value - v_min) / (v_max - v_min)
            x = x_rel * rect.width
            y = rect.height - y_rel * rect.height
            return int(round(max(0.0, min(rect.width, x)))), int(round(max(0.0, min(rect.height, y))))

        points: list[tuple[int, int]] = []
        prev_time = times[0]
        prev_val = values[0]
        points.append(to_point(prev_time, prev_val))
        for time_value, flux_value in zip(times[1:], values[1:]):
            points.append(to_point(time_value, prev_val))
            points.append(to_point(time_value, flux_value))
            prev_time = time_value
            prev_val = flux_value

        if len(points) >= 2:
            pygame.draw.lines(graph_surface, line_color, False, points, 2)

        target_surface.blit(graph_surface, rect.topleft)

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
        self.reset_wall_hit_counters()
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
        self._ensure_tracked_particle()
        if self.trail_enabled:
            self.trail_points.clear()
        # Update parameter dictionary to reflect increased number of particles
        if 'r' in self.params:
            self.params['r'] += count

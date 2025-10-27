"""
Updated DemoScreen without energy graphs.

This version of ``demo_screen.py`` removes all chart (graph) objects that
previously displayed kinetic and potential energies of spring particles.
Only the demonstration of the gas particles in the box remains, along
with sliders and control buttons.  Methods related to charts are
retained but reduced to no‑ops to preserve API compatibility.
"""

import math
from collections import OrderedDict

import pygame
import language
from button import Button
from slider import ParamSlider
from demo import Demo
import config


class DemoScreen:
    def __init__(self, app):
        lang = language.Language()
        self.app = app
        self.screen = app.screen
        self.speed = 0.5
        self.bg_color = (210, 210, 210)
        self.font = 'corbel'
        self.little_font = pygame.font.SysFont(self.font, 35)
        self.middle_font = pygame.font.SysFont(self.font, 40, bold=True)
        self.big_font = pygame.font.SysFont(self.font, 50)

        # Layout parameters for the full-width simulation box and the controls row
        side_margin = 30
        top_margin = 40
        screen_width = int(self.app.monitor.width * 0.85)
        screen_height = int(self.app.monitor.height * 0.85)
        # Resize the display surface to the reduced dimensions so the layout is predictable
        self.app.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen = self.app.screen

        # Load parameter definitions from configuration
        param_names, _, _, param_bounds, param_initial, param_step, par4sim, dec_numbers = self._load_params()
        # Preserve the original param bounds for later use (e.g. to size data arrays)
        _original_param_bounds = list(param_bounds)
        # Normalize initial values into [0, 1] interval
        param_initial = list(map(_init_val_into_unit, param_initial, param_bounds))
        # Remove sliders corresponding to parameters that are unused in this
        # simplified model (size of connected particles, elasticity, nonlinearity
        # coefficient and mass of connected particles).  The underlying
        # simulation parameters for these sliders are identified by the
        # entries in ``par4sim``.  When the name_par belongs to the
        # ignore set the slider is omitted.
        ignore_params = {'gamma', 'k', 'm_spring', 'R', 'R_spring'}
        entries = list(zip(param_names, param_bounds, param_initial, param_step, par4sim, dec_numbers))
        filtered_entries = [e for e in entries if e[4] not in ignore_params]
        if filtered_entries:
            param_names, param_bounds, param_initial, param_step, par4sim, dec_numbers = zip(*filtered_entries)
            param_names = list(param_names)
            param_bounds = list(param_bounds)
            param_initial = list(param_initial)
            param_step = list(param_step)
            par4sim = list(par4sim)
            dec_numbers = list(dec_numbers)
        else:
            # No parameters remain after filtering
            param_names, param_bounds, param_initial, param_step, par4sim, dec_numbers = ([], [], [], [], [], [])

        # Collect slider definitions (label, bounds, key, etc.)
        slider_defs = [
            {
                'label': name,
                'bounds': bounds,
                'initial_pos': initial,
                'step': step,
                'key': name_par,
                'decimals': dec_number,
            }
            for name, bounds, initial, step, name_par, dec_number in zip(param_names, param_bounds, param_initial, param_step, par4sim, dec_numbers)
        ]

        # Additional sliders for thermal wall parameters (T_left, T_right) and
        # accommodation coefficient.  These sliders allow the user to adjust
        # the temperatures of the left and right walls and the degree of
        # thermal accommodation (0 = purely specular reflection, 1 = full
        # thermalization).
        min_T = 100.0
        max_T = 1000.0
        initial_T_left = 600.0
        init_pos_T_left = (initial_T_left - min_T) / (max_T - min_T)
        step_T = (max_T - min_T) / 100.0
        initial_T_right = 300.0
        init_pos_T_right = (initial_T_right - min_T) / (max_T - min_T)
        min_accom = 0.0
        max_accom = 1.0
        initial_accom = 1.0
        init_pos_accom = (initial_accom - min_accom) / (max_accom - min_accom)
        step_accom = (max_accom - min_accom) / 100.0
        slider_defs.extend(
            [
                {
                    'label': 'T_left',
                    'bounds': (min_T, max_T),
                    'initial_pos': init_pos_T_left,
                    'step': step_T,
                    'key': 'T_left',
                    'decimals': 0,
                },
                {
                    'label': 'T_right',
                    'bounds': (min_T, max_T),
                    'initial_pos': init_pos_T_right,
                    'step': step_T,
                    'key': 'T_right',
                    'decimals': 0,
                },
                {
                    'label': 'accommodation',
                    'bounds': (min_accom, max_accom),
                    'initial_pos': init_pos_accom,
                    'step': step_accom,
                    'key': 'accommodation',
                    'decimals': 2,
                },
            ]
        )

        # Additional sliders for particle size and number of tagged particles.
        # ``slider_size_scale`` controls both the physical and visual radius
        # of gas particles simultaneously.  ``slider_tag_count`` sets the
        # number of marked particles to add when the "Add particles" button
        # is pressed.
        size_bounds = (0.5, 1.5)
        size_initial = 1.3
        init_pos_size = (size_initial - size_bounds[0]) / (size_bounds[1] - size_bounds[0])
        step_size = (size_bounds[1] - size_bounds[0]) / 100.0
        tag_bounds = (1, 100)
        tag_initial = 10  # default number of tagged particles
        init_pos_tag = (tag_initial - tag_bounds[0]) / (tag_bounds[1] - tag_bounds[0])
        step_tag = 1
        slider_defs.extend(
            [
                {
                    'label': 'size_scale',
                    'bounds': size_bounds,
                    'initial_pos': init_pos_size,
                    'step': step_size,
                    'key': 'size_scale',
                    'decimals': 2,
                },
                {
                    'label': 'tagged_count',
                    'bounds': tag_bounds,
                    'initial_pos': init_pos_tag,
                    'step': step_tag,
                    'key': 'tagged_count',
                    'decimals': 0,
                },
            ]
        )

        slowmo_bounds = (0.1, 1.0)
        slowmo_initial = 1.0
        init_pos_slowmo = (slowmo_initial - slowmo_bounds[0]) / (slowmo_bounds[1] - slowmo_bounds[0])
        step_slowmo = (slowmo_bounds[1] - slowmo_bounds[0]) / 100.0
        slider_defs.append(
            {
                'label': 'Слоумо',
                'bounds': slowmo_bounds,
                'initial_pos': init_pos_slowmo,
                'step': step_slowmo,
                'key': 'slowmo',
                'decimals': 2,
            }
        )

        total_sliders = len(slider_defs)
        if total_sliders:
            controls_rows = 2 if total_sliders > 2 else total_sliders
            controls_cols = math.ceil(total_sliders / controls_rows)
        else:
            controls_rows = 0
            controls_cols = 1

        button_height = 50
        button_area_height = 75   # includes padding around buttons
        min_sim_height = 200
        slider_area_height = max(140, controls_rows * 80) if controls_rows else 140
        controls_height = button_area_height + slider_area_height
        max_controls_height = screen_height - (top_margin + min_sim_height)
        if max_controls_height < button_area_height + 100:
            max_controls_height = button_area_height + 100
        controls_height = min(controls_height, max_controls_height)
        controls_top = screen_height - controls_height
        base_sim_height = screen_height - controls_height - top_margin - 10
        sim_height = max(min_sim_height, int(base_sim_height * 0.8))
        controls_top = top_margin + sim_height + 10
        controls_height = screen_height - controls_top
        min_controls_height = button_area_height + 70
        if controls_height < min_controls_height:
            deficit = min_controls_height - controls_height
            sim_height = max(min_sim_height, sim_height - deficit)
            controls_top = top_margin + sim_height + 10
            controls_height = screen_height - controls_top

        sim_width = screen_width - 2 * side_margin

        # Buttons: Add Particles, Apply changes, Menu. Mode button remains removed.
        btn_w, btn_h = 190, button_height
        btn_gap = 16
        btn_y = controls_top + 8
        add_label = 'Добавить частицы'
        apply_label = 'Применить'
        menu_label = 'Меню'
        dim_label = 'Затусклить фон'
        trail_label = 'Показать след'
        button_specs = [
            (add_label, self.add_marked_particles, None),
            (dim_label, self.toggle_dim_particles, 'dim_button'),
            (trail_label, self.toggle_trail, 'trail_button'),
            (apply_label, self.apply, None),
            (menu_label, self.to_menu, None),
        ]
        self.dim_button = None
        self.trail_button = None
        self.buttons = []
        for i, (label, handler, attr_name) in enumerate(button_specs):
            btn = Button(
                app,
                label,
                (side_margin + i * (btn_w + btn_gap), btn_y),
                (btn_w, btn_h),
                handler,
                font='sans',
                bold=False,
                fontSize=24,
            )
            self.buttons.append(btn)
            if attr_name == 'dim_button':
                self.dim_button = btn
            elif attr_name == 'trail_button':
                self.trail_button = btn

        # Compute slider positions in a grid beneath the buttons
        slider_start_y = btn_y + btn_h + 15
        slider_area_bottom = controls_top + controls_height - 10
        available_slider_height = max(40.0, slider_area_bottom - slider_start_y)
        slider_positions = []
        if controls_rows:
            row_spacing = available_slider_height / controls_rows
        else:
            row_spacing = available_slider_height
        controls_width = screen_width - 2 * side_margin
        col_spacing = controls_width / controls_cols if controls_cols else controls_width
        for idx in range(total_sliders):
            row = idx // controls_cols if controls_cols else 0
            col = idx % controls_cols if controls_cols else 0
            cx = side_margin + col_spacing * (col + 0.5)
            cy = slider_start_y + row_spacing * (row + 0.5)
            slider_positions.append((int(cx), int(cy)))

        # Instantiate sliders and keep ordered mapping to parameter keys
        self.sliders: list[ParamSlider] = []
        self.slider_param_keys: list[str] = []
        self.slider_T_left = None
        self.slider_T_right = None
        self.slider_accommodation = None
        self.slider_size_scale = None
        self.slider_tag_count = None
        self.slider_slowmo = None

        for index, (definition, position) in enumerate(zip(slider_defs, slider_positions)):
            slider = ParamSlider(
                app,
                definition['label'],
                position,
                definition['bounds'],
                definition['step'],
                definition['key'],
                definition['decimals'],
                button_color=self.bg_color,
                font='sans',
                bold=False,
                fontSize=18,
                initial_pos=definition['initial_pos'],
            )
            self.sliders.append(slider)
            self.slider_param_keys.append(definition['key'])

            if definition['key'] == 'T_left':
                self.slider_T_left = slider
            elif definition['key'] == 'T_right':
                self.slider_T_right = slider
            elif definition['key'] == 'accommodation':
                self.slider_accommodation = slider
            elif definition['key'] == 'size_scale':
                self.slider_size_scale = slider
            elif definition['key'] == 'tagged_count':
                self.slider_tag_count = slider
            elif definition['key'] == 'slowmo':
                self.slider_slowmo = slider

        # Ensure we have references for mandatory sliders
        assert self.slider_T_left is not None and self.slider_T_right is not None
        assert self.slider_accommodation is not None
        assert self.slider_size_scale is not None and self.slider_tag_count is not None
        assert self.slider_slowmo is not None
        sim_width = max(100, int(round(sim_width)))
        sim_height = max(100, int(round(sim_height)))

        # Initialize the demonstration with current slider values
        params_initial = OrderedDict()
        for key, slider in zip(self.slider_param_keys, self.sliders):
            params_initial[key] = slider.getValue()

        self.demo = Demo(
            app,
            (int(side_margin), int(top_margin)),
            (sim_width, sim_height),
            (255, 255, 255),
            (100, 100, 100),
            self.bg_color,
            params_initial.copy(),
        )
        self.dim_active = False
        self.trail_active = False
        self.demo.set_dim_untracked(self.dim_active)
        self.demo.set_trail_enabled(self.trail_active)

        # Demo configuration used by charts and sliders.  Note that the
        # number of elements in the data arrays (e.g. kinetic) is
        # derived from the original parameter bounds prior to filtering,
        # because the filtered ``param_bounds`` no longer necessarily
        # includes the parameter that defined the intended length.
        frames_count = _original_param_bounds[-1][1] if _original_param_bounds else 10
        self.demo_config = {
            'params': params_initial.copy(),
            'kinetic': [0] * frames_count,
            'mean_kinetic': [0] * frames_count,
            'potential': [0] * frames_count,
            'mean_potential': [0] * frames_count,
            'is_changed': False,
        }
        # No charts are created in this version
        self.graphics = []
        self.slider_grabbed = False
        self.charts_mode = False  # mode toggle unused

        # Number of tagged particles to add when the user presses the Add button.
        self.tagged_count = int(params_initial.get('tagged_count', 10))

    # ---------------------------------------------------------------------
    def correct_limits(self):
        """Placeholder for chart y‑limit synchronization (no‑op)."""
        return

    def apply(self):
        """Refresh simulation and (formerly) charts when Apply button pressed."""
        # Update simulation with thermal parameters from sliders
        self.demo.simulation.set_params(
            T_left=self.slider_T_left.getValue(),
            T_right=self.slider_T_right.getValue(),
            accommodation=self.slider_accommodation.getValue(),
        )
        # Update params in demo_config
        self.demo_config['params']['T_left'] = self.slider_T_left.getValue()
        self.demo_config['params']['T_right'] = self.slider_T_right.getValue()
        self.demo_config['params']['accommodation'] = self.slider_accommodation.getValue()
        # Apply particle size scale only when the user presses Apply.  Both
        # the physical and visual radii will be updated together.  The
        # value comes from the size slider; cast to float to avoid
        # issues when the slider returns numpy types.
        size_val = float(self.slider_size_scale.getValue())
        self.demo.update_radius_scale(size_val)
        self.demo_config['params']['size_scale'] = size_val
        # Update the number of tagged particles to add next time the
        # Add button is pressed.  Cast to int to ensure discrete count.
        self.tagged_count = int(self.slider_tag_count.getValue())
        self.demo_config['params']['tagged_count'] = self.tagged_count
        slowmo_val = float(self.slider_slowmo.getValue())
        self.demo.set_time_scale(slowmo_val)
        self.demo_config['params']['slowmo'] = slowmo_val
        # Without charts, just refresh the simulation.  ``_refresh_iter``
        # will update changed parameters (like number of particles) via
        # ``Demo.set_params`` if needed.
        self.demo._refresh_iter(self.demo_config)
        self.demo_config['is_changed'] = False

    def add_marked_particles(self):
        """Callback for the Add Particles button.

        Invokes ``Demo.add_tagged_particles`` with the configured number
        of tagged particles and updates the particle count parameter.
        """
        # Add the tagged particles to the simulation
        self.demo.add_tagged_particles(self.tagged_count)
        # Update the stored parameter for number of particles (if present)
        # Note: some slider names may map to different keys; adjust as needed.
        if 'r' in self.demo_config['params']:
            self.demo_config['params']['r'] += self.tagged_count
        # Mark the config as changed so that the next refresh recomputes values
        self.demo_config['is_changed'] = True

    def toggle_dim_particles(self):
        """Toggle dimming of untagged particles for easier tracking."""
        self.dim_active = not self.dim_active
        self.demo.set_dim_untracked(self.dim_active)
        if self.dim_button is not None:
            new_label = 'Вернуть цвета' if self.dim_active else 'Затусклить фон'
            self.dim_button._prep_msg(new_label)

    def toggle_trail(self):
        """Toggle trajectory drawing for the highlighted tagged particle."""
        self.trail_active = not self.trail_active
        self.demo.set_trail_enabled(self.trail_active)
        if self.trail_button is not None:
            new_label = 'Скрыть след' if self.trail_active else 'Показать след'
            self.trail_button._prep_msg(new_label)

    def modes(self):
        """Toggle chart display mode (no‑op)."""
        # No charts to toggle
        return

    def to_menu(self):
        self.app.active_screen = self.app.menu_screen

    def _load_params(self):
        loader = config.ConfigLoader()
        lang = language.Language()
        param_names = [lang[name] for name in loader['param_names']]
        sliders_gap = loader['sliders_gap']
        param_poses = [(self.app.monitor.width * 0.82 + 40, h) for h in range(50, 150 + len(param_names) * sliders_gap + 1, sliders_gap)]
        param_bounds = []
        param_initial = []
        for param_name in loader['param_names']:
            param_bounds.append(tuple(loader['param_bounds'][param_name]))
            param_initial.append(loader['param_initial'][param_name])
        param_step = [round((b[1] - b[0]) / 100, 3) for b in param_bounds]
        # Steps for discrete parameters should be integers
        param_step[1], param_step[2] = int(param_step[1]), int(param_step[2])
        par4sim = loader['par4sim']
        dec_numbers = [1, 0, 0, 0, 1, 0, 0]
        return param_names, sliders_gap, param_poses, param_bounds, param_initial, param_step, par4sim, dec_numbers

    # ---------------------------------------------------------------------
    def _update_screen(self):
        self.screen.fill(self.bg_color)
        self.demo.draw_check(self.demo_config)
        self._draw_counters()
        for button in self.buttons:
            button.draw_button()
        for slider in self.sliders:
            slider.draw_check(self.demo_config['params'])
        # No charts drawn
        # self._draw_figures()  # removed

    def _check_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                self._check_buttons(mouse_position)
            mouse_pos = pygame.mouse.get_pos()
            mouse = pygame.mouse.get_pressed()
            self._check_sliders(mouse_pos, mouse)

    def _check_sliders(self, mouse_position, mouse_pressed):
        for slider in self.sliders:
            if slider.slider.button_rect.collidepoint(mouse_position):
                if mouse_pressed[0] and not self.slider_grabbed:
                    slider.slider.grabbed = True
                    self.slider_grabbed = True
            if not mouse_pressed[0]:
                slider.slider.grabbed = False
                self.slider_grabbed = False
            if slider.slider.button_rect.collidepoint(mouse_position):
                slider.slider.hover()
            if slider.slider.grabbed:
                slider.slider.move_slider(mouse_position)
                slider.slider.hover()
            else:
                slider.slider.hovered = False

    def _check_buttons(self, mouse_position):
        for button in self.buttons:
            if button.rect.collidepoint(mouse_position):
                button.command()

    def _draw_figures(self):
        """Draw charts if any exist (no charts in this version)."""
        for fig in self.graphics:
            fig.draw(self.demo_config)
        # Synchronize y‑limits (unused)
        self.correct_limits()

    def _draw_counters(self):
        """Render counters showing how many tagged particles hit each wall."""
        if not self.demo.has_tagged_particles():
            return
        left_hits, right_hits = self.demo.get_wall_hit_counts()
        left_text = f'Левая: {left_hits}'
        right_text = f'Правая: {right_hits}'
        left_surface = self.middle_font.render(left_text, True, (220, 70, 40))
        right_surface = self.middle_font.render(right_text, True, (60, 130, 255))
        padding = 10
        top_y = self.demo.position[1] + padding
        left_x = self.demo.position[0] + padding
        right_x = self.demo.position[0] + self.demo.width - right_surface.get_width() - padding
        self.screen.blit(left_surface, (left_x, top_y))
        self.screen.blit(right_surface, (right_x, top_y))


def _init_val_into_unit(initial_val, bounds) -> float:
    if not (bounds[0] <= initial_val <= bounds[1]):
        raise ValueError("Initial val mus be in [bounds[0], bounds[1]]")
    return (initial_val - bounds[0]) / (bounds[1] - bounds[0])

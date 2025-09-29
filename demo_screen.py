"""
Updated DemoScreen without energy graphs.

This version of ``demo_screen.py`` removes all chart (graph) objects that
previously displayed kinetic and potential energies of spring particles.
Only the demonstration of the gas particles in the box remains, along
with sliders and control buttons.  Methods related to charts are
retained but reduced to no‑ops to preserve API compatibility.
"""

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

        # Buttons: Add Particles, Apply changes, Menu.  Mode button has been
        # removed as there is no alternate display mode.  Button labels
        # are provided directly in Russian rather than relying on the
        # language dictionary.
        base_btn_x = app.monitor.width * 0.05 + 30
        btn_y = app.monitor.width * 0.43 + 60
        btn_w, btn_h = 250, 80
        # Russian labels for buttons
        add_label = 'Добавить частицы'
        apply_label = 'Применить'
        menu_label = 'Меню'
        # Place buttons left‑to‑right: Add → Apply → Menu
        self.buttons = [
            Button(app, add_label, (base_btn_x + 0 * 290, btn_y), (btn_w, btn_h), self.add_marked_particles),
            Button(app, apply_label, (base_btn_x + 1 * 290, btn_y), (btn_w, btn_h), self.apply),
            Button(app, menu_label, (base_btn_x + 2 * 290, btn_y), (btn_w, btn_h), self.to_menu),
        ]

        # Load parameter definitions from configuration
        param_names, sliders_gap, param_poses, param_bounds, param_initial, param_step, par4sim, dec_numbers = self._load_params()
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
        entries = list(zip(param_names, param_poses, param_bounds, param_initial, param_step, par4sim, dec_numbers))
        filtered_entries = [e for e in entries if e[5] not in ignore_params]
        if filtered_entries:
            param_names, param_poses, param_bounds, param_initial, param_step, par4sim, dec_numbers = zip(*filtered_entries)
            param_names = list(param_names)
            param_poses = list(param_poses)
            param_bounds = list(param_bounds)
            param_initial = list(param_initial)
            param_step = list(param_step)
            par4sim = list(par4sim)
            dec_numbers = list(dec_numbers)
        else:
            # No parameters remain after filtering
            param_names, param_poses, param_bounds, param_initial, param_step, par4sim, dec_numbers = ([], [], [], [], [], [], [])
        # Create sliders only for the kept simulation parameters
        self.sliders = [
            ParamSlider(
                app,
                name,
                pos,
                bounds,
                step,
                name_par,
                dec_number,
                button_color=self.bg_color,
                font='sans',
                bold=False,
                fontSize=32,
                initial_pos=initial,
            )
            for name, pos, bounds, initial, step, name_par, dec_number in zip(param_names, param_poses, param_bounds, param_initial, param_step, par4sim, dec_numbers)
        ]

        # -----------------------------------------------------------------
        # Additional sliders for thermal wall parameters (T_left, T_right) and
        # accommodation coefficient.  These sliders allow the user to adjust
        # the temperatures of the left and right walls and the degree of
        # thermal accommodation (0 = purely specular reflection, 1 = full
        # thermalization).  Positions are determined relative to the last
        # existing slider and the configured gap.
        base_x = param_poses[0][0]
        base_y = param_poses[-1][1] + sliders_gap

        # Define bounds and initial values for temperatures (in kelvins)
        min_T = 100.0
        max_T = 1000.0
        # Left wall temperature slider
        initial_T_left = 600.0
        init_pos_T_left = (initial_T_left - min_T) / (max_T - min_T)
        step_T = (max_T - min_T) / 100.0
        self.slider_T_left = ParamSlider(
            app,
            'T_left',
            (base_x, base_y),
            (min_T, max_T),
            step_T,
            'T_left',
            0,
            button_color=self.bg_color,
            font='sans',
            bold=False,
            fontSize=32,
            initial_pos=init_pos_T_left,
        )
        # Right wall temperature slider
        initial_T_right = 300.0
        init_pos_T_right = (initial_T_right - min_T) / (max_T - min_T)
        self.slider_T_right = ParamSlider(
            app,
            'T_right',
            (base_x, base_y + sliders_gap),
            (min_T, max_T),
            step_T,
            'T_right',
            0,
            button_color=self.bg_color,
            font='sans',
            bold=False,
            fontSize=32,
            initial_pos=init_pos_T_right,
        )
        # Accommodation coefficient slider (0 – 1)
        min_accom = 0.0
        max_accom = 1.0
        initial_accom = 1.0
        init_pos_accom = (initial_accom - min_accom) / (max_accom - min_accom)
        step_accom = (max_accom - min_accom) / 100.0
        self.slider_accommodation = ParamSlider(
            app,
            'accommodation',
            (base_x, base_y + 2 * sliders_gap),
            (min_accom, max_accom),
            step_accom,
            'accommodation',
            2,
            button_color=self.bg_color,
            font='sans',
            bold=False,
            fontSize=32,
            initial_pos=init_pos_accom,
        )
        # Append thermal sliders to the main sliders list
        self.sliders.extend([self.slider_T_left, self.slider_T_right, self.slider_accommodation])

        # -----------------------------------------------------------------
        # Additional sliders for particle size and number of tagged particles.
        # ``slider_size_scale`` controls both the physical and visual radius
        # of gas particles simultaneously.  ``slider_tag_count`` sets the
        # number of marked particles to add when the "Add particles" button
        # is pressed.  These controls are placed below the thermal sliders.
        # Slider bounds for particle size scaling (50%–150%).  The user
        # wanted the upper limit reduced to 1.5 instead of 3.0, and
        # default particles of size 1.0.
        size_bounds = (0.5, 1.5)
        size_initial = 1.0
        init_pos_size = (size_initial - size_bounds[0]) / (size_bounds[1] - size_bounds[0])
        step_size = (size_bounds[1] - size_bounds[0]) / 100.0
        # Slider controlling particle size (both physical and visual)
        self.slider_size_scale = ParamSlider(
            app,
            'size_scale',
            (base_x, base_y + 3 * sliders_gap),
            size_bounds,
            step_size,
            'size_scale',
            2,
            button_color=self.bg_color,
            font='sans',
            bold=False,
            fontSize=32,
            initial_pos=init_pos_size,
        )
        # Bounds for number of tagged particles to add
        tag_bounds = (1, 100)
        tag_initial = 10  # default number of tagged particles
        init_pos_tag = (tag_initial - tag_bounds[0]) / (tag_bounds[1] - tag_bounds[0])
        step_tag = 1
        self.slider_tag_count = ParamSlider(
            app,
            'tagged_count',
            (base_x, base_y + 4 * sliders_gap),
            tag_bounds,
            step_tag,
            'tagged_count',
            0,
            button_color=self.bg_color,
            font='sans',
            bold=False,
            fontSize=32,
            initial_pos=init_pos_tag,
        )
        # Add the size and tag count sliders to the list
        self.sliders.extend([self.slider_size_scale, self.slider_tag_count])


        # Initialize the demonstration with current slider values
        self.demo = Demo(
            app,
            (app.monitor.width * 0.05 + 30, 30),
            (app.monitor.width * 0.43, app.monitor.width * 0.43),
            (255, 255, 255),
            (100, 100, 100),
            self.bg_color,
            {name: sl.getValue() for name, sl in zip(par4sim, self.sliders)},
        )

        # Demo configuration used by charts and sliders.  Note that the
        # number of elements in the data arrays (e.g. kinetic) is
        # derived from the original parameter bounds prior to filtering,
        # because the filtered ``param_bounds`` no longer necessarily
        # includes the parameter that defined the intended length.
        frames_count = _original_param_bounds[-1][1] if _original_param_bounds else 10
        self.demo_config = {
            'params': {name: sl.getValue() for name, sl in zip(par4sim, self.sliders)},
            'kinetic': [0] * frames_count,
            'mean_kinetic': [0] * frames_count,
            'potential': [0] * frames_count,
            'mean_potential': [0] * frames_count,
            'is_changed': False,
        }
        # Add thermal parameters (T_left, T_right, accommodation) to params dict
        # Their values come from the sliders created above
        self.demo_config['params']['T_left'] = self.slider_T_left.getValue()
        self.demo_config['params']['T_right'] = self.slider_T_right.getValue()
        self.demo_config['params']['accommodation'] = self.slider_accommodation.getValue()

        # Add size scale and tagged count parameters to the params dict.
        # These values are stored for later application but do not directly
        # affect the simulation until the user presses Apply.
        self.demo_config['params']['size_scale'] = self.slider_size_scale.getValue()
        self.demo_config['params']['tagged_count'] = self.slider_tag_count.getValue()
        # No charts are created in this version
        self.graphics = []
        self.slider_grabbed = False
        self.charts_mode = False  # mode toggle unused

        # Number of tagged particles to add when the user presses the Add button.
        # Default changed to 10 in accordance with the requirements.
        self.tagged_count: int = 10

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


def _init_val_into_unit(initial_val, bounds) -> float:
    if not (bounds[0] <= initial_val <= bounds[1]):
        raise ValueError("Initial val mus be in [bounds[0], bounds[1]]")
    return (initial_val - bounds[0]) / (bounds[1] - bounds[0])
from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Tuple

import pygame

import language
from button import Button
from slider import ParamSlider
from demo import Demo
import config
from ui_base import ResponsiveScreen, get_font, build_vertical_gradient


class DemoScreen(ResponsiveScreen):
    def __init__(self, app):
        super().__init__(app)
        self.lang = language.Language()
        self.primary_color = (72, 104, 255)
        self.bg_color = (234, 236, 246)
        self.panel_color = (248, 249, 253)
        self.border_color = (214, 220, 235)
        self.shadow_color = (18, 24, 60, 60)

        self.counter_font = get_font(30, bold=True)
        self.panel_title_font = get_font(24, bold=True)
        self.placeholder_font = get_font(20)
        self.graph_value_font = get_font(22, bold=True)
        self.graph_counts_font = get_font(18)
        self.graph_max_samples = 600

        self.slider_panel_rect: pygame.Rect | None = None
        self.right_panel_rect: pygame.Rect | None = None
        self.sim_rect: pygame.Rect | None = None

        self.background: pygame.Surface | None = None

        self.buttons: List[Button] = []
        self.dim_button: Button | None = None
        self.trail_button: Button | None = None
        self.add_button: Button | None = None
        self.apply_button: Button | None = None
        self.menu_button: Button | None = None

        self.graph_value_template: str = ''
        self.graph_counts_template: str = ''

        self._localize_strings()
        self._build_button_layout()

        self.sliders: List[ParamSlider] = []
        self.slider_param_keys: List[str] = []
        self.slider_T_left = None
        self.slider_T_right = None
        self.slider_accommodation = None
        self.slider_size_scale = None
        self.slider_tag_count = None
        self.slider_slowmo = None

        self.demo: Demo | None = None
        self.demo_config = {}
        self.dim_active = False
        self.trail_active = False
        self.slider_grabbed = False
        self.tagged_count = 10

        (
            self.slider_definitions,
            original_bounds,
            self.initial_params,
        ) = self._build_slider_definitions()

        frames_count = original_bounds[-1][1] if original_bounds else 10
        self.demo_config = {
            'params': self.initial_params.copy(),
            'kinetic': [0] * frames_count,
            'mean_kinetic': [0] * frames_count,
            'potential': [0] * frames_count,
            'mean_potential': [0] * frames_count,
            'is_changed': False,
        }
        self.tagged_count = int(self.initial_params.get('tagged_count', 10))

        self.background = build_vertical_gradient(self.app.window_size, (230, 236, 255), (246, 248, 254))
        self._relayout(self.app.window_size)

    def on_language_change(self) -> None:
        self.lang = language.Language()
        self._localize_strings()
        self._build_button_layout()
        current_values = self.demo_config.get('params', {}).copy()
        self.slider_definitions, _, self.initial_params = self._build_slider_definitions()
        for key, value in current_values.items():
            if key in self.initial_params:
                self.initial_params[key] = value
        self.demo_config['params'].update(self.initial_params)
        self.tagged_count = int(self.demo_config['params'].get('tagged_count', self.tagged_count))
        self._relayout(self.app.window_size)
        if self.dim_button is not None:
            dim_label = self.dim_labels[1] if self.dim_active else self.dim_labels[0]
            self.dim_button._prep_msg(dim_label)
        if self.trail_button is not None:
            trail_label = self.trail_labels[1] if self.trail_active else self.trail_labels[0]
            self.trail_button._prep_msg(trail_label)

    def _build_button_layout(self) -> None:
        apply_label = self.lang['btn_apply']
        menu_label = self.lang['btn_menu']
        self.button_layout = [
            {'label': self.label_add_particles, 'handler': self.add_marked_particles, 'primary': True, 'attr': 'add_button'},
            {'label': self.dim_labels[0], 'handler': self.toggle_dim_particles, 'id': 'dim'},
            {'label': self.trail_labels[0], 'handler': self.toggle_trail, 'id': 'trail'},
            {'label': apply_label, 'handler': self.apply, 'attr': 'apply_button'},
            {'label': menu_label, 'handler': self.to_menu, 'attr': 'menu_button'},
        ]

    def _localize_strings(self) -> None:
        def _fallback(key: str, default: str) -> str:
            try:
                return self.lang[key]
            except KeyError:
                return default

        if self.lang.lang == 'rus':
            default_add = 'Добавить частицы'
            dim_pair = ('Затусклить фон', 'Вернуть цвета')
            trail_pair = ('Показать след', 'Скрыть след')
            panel = 'Параметры'
            graphs = 'Поток через стену'
            placeholder = 'Недостаточно данных для графика'
            counter_left = 'Левая'
            counter_right = 'Правая'
            flux_value_default = 'Поток: {value:+.1f} част./ед. времени'
            counts_default = '-> {forward}   <- {backward}'
        else:
            default_add = 'Add particles'
            dim_pair = ('Dim background', 'Restore colors')
            trail_pair = ('Show trail', 'Hide trail')
            panel = 'Parameters'
            graphs = 'Midplane flux'
            placeholder = 'Not enough data yet'
            counter_left = 'Left'
            counter_right = 'Right'
            flux_value_default = 'Flux: {value:+.1f} particles/unit time'
            counts_default = '-> {forward}   <- {backward}'

        self.label_add_particles = _fallback('btn_add_particles', default_add)
        dim_enable = _fallback('btn_dim_enable', dim_pair[0])
        dim_disable = _fallback('btn_dim_disable', dim_pair[1])
        trail_enable = _fallback('btn_trail_enable', trail_pair[0])
        trail_disable = _fallback('btn_trail_disable', trail_pair[1])
        self.dim_labels = (dim_enable, dim_disable)
        self.trail_labels = (trail_enable, trail_disable)
        self.panel_title_text = _fallback('panel_title_parameters', panel)
        self.graph_title_text = _fallback('panel_title_charts', graphs)
        self.placeholder_text = _fallback('charts_placeholder', placeholder)
        value_template = _fallback('flux_value_label', flux_value_default)
        if '{value' not in value_template:
            value_template = flux_value_default
        counts_template = _fallback('flux_counts_label', counts_default)
        if '{forward' not in counts_template or '{backward' not in counts_template:
            counts_template = counts_default
        self.graph_value_template = value_template
        self.graph_counts_template = counts_template
        default_left_template = f'{counter_left}: {{count}}'
        default_right_template = f'{counter_right}: {{count}}'
        left_value = _fallback('counter_left', default_left_template)
        right_value = _fallback('counter_right', default_right_template)
        self.counter_left_template = left_value if '{count}' in left_value else f'{left_value}: {{count}}'
        self.counter_right_template = right_value if '{count}' in right_value else f'{right_value}: {{count}}'

    # ------------------------------------------------------------------ Layout
    def _relayout(self, size: tuple[int, int]) -> None:
        width, height = size
        width = max(width, 1200)
        height = max(height, 720)
        self.background = build_vertical_gradient((width, height), (230, 236, 255), (246, 248, 254))

        margin = max(28, width // 48)
        sim_height = max(240, int(height * 0.42))
        self.sim_rect = pygame.Rect(margin, margin, width - 2 * margin, sim_height)

        bottom_top = self.sim_rect.bottom + 30
        bottom_height = max(260, height - bottom_top - margin)
        available_width = width - 2 * margin
        left_panel_width = max(460, int(available_width * 0.55))
        right_panel_width = available_width - left_panel_width - margin
        if right_panel_width < 320:
            left_panel_width = available_width
            right_panel_width = 0
        self.slider_panel_rect = pygame.Rect(margin, bottom_top, left_panel_width, bottom_height)
        if right_panel_width > 0:
            right_x = self.slider_panel_rect.right + margin
            self.right_panel_rect = pygame.Rect(right_x, bottom_top, right_panel_width, bottom_height)
        else:
            self.right_panel_rect = None

        self._build_buttons()
        self._build_sliders()
        self._ensure_demo()

    def _build_buttons(self) -> None:
        assert self.slider_panel_rect is not None
        panel = self.slider_panel_rect
        inner_padding = max(18, panel.width // 26)
        gap = max(10, inner_padding // 3)
        button_height = max(42, min(52, panel.height // 6))
        layout_rows = [self.button_layout[:3], self.button_layout[3:]]

        self.buttons = []
        self.dim_button = None
        self.trail_button = None
        self.add_button = None
        self.apply_button = None
        self.menu_button = None
        rows_total = len(layout_rows)
        total_height = rows_total * button_height + (rows_total - 1) * gap
        start_y = panel.bottom - inner_padding - total_height
        current_y = start_y
        self.button_area_top = start_y
        for row in layout_rows:
            cols = len(row)
            row_width_available = panel.width - 2 * inner_padding
            button_width = (row_width_available - gap * (cols - 1)) / cols
            button_width = max(150, min(240, button_width))
            total_width = cols * button_width + (cols - 1) * gap
            start_x = panel.left + (panel.width - total_width) / 2
            for spec in row:
                button = Button(
                    self.app,
                    spec['label'],
                    (int(start_x), int(current_y)),
                    (int(button_width), button_height),
                    spec['handler'],
                    font=["SF Pro Display", "Segoe UI", "Arial"],
                    fontSize=max(20, int(button_height * 0.4)),
                    bold=True,
                    button_color=self.primary_color if spec.get('primary') else (249, 250, 253),
                    text_color=(255, 255, 255) if spec.get('primary') else (35, 38, 46),
                    border_radius=16,
                    border_color=None if spec.get('primary') else self.border_color,
                    shadow_offset=8 if spec.get('primary') else 5,
                    shadow_color=(64, 99, 255, 120) if spec.get('primary') else (0, 0, 0, 45),
                )
                self.buttons.append(button)
                if spec.get('id') == 'dim':
                    self.dim_button = button
                elif spec.get('id') == 'trail':
                    self.trail_button = button
                attr_name = spec.get('attr')
                if attr_name == 'add_button':
                    self.add_button = button
                elif attr_name == 'apply_button':
                    self.apply_button = button
                elif attr_name == 'menu_button':
                    self.menu_button = button
            start_x += button_width + gap
            current_y += button_height + gap
        self.button_area_bottom = current_y - gap

    def _build_sliders(self) -> None:
        assert self.slider_panel_rect is not None
        panel = self.slider_panel_rect
        inner_padding = max(20, panel.width // 28)
        gap = max(12, inner_padding // 3)
        title_height = self.panel_title_font.get_height()
        content_top = panel.top + inner_padding + title_height + max(12, inner_padding // 3)
        buttons_top = getattr(self, 'button_area_top', panel.bottom - inner_padding)
        content_bottom = buttons_top - max(14, inner_padding // 3)
        if content_bottom <= content_top:
            content_top = panel.top + inner_padding + title_height
            content_bottom = panel.bottom - inner_padding
        available_height = max(40.0, content_bottom - content_top)

        total_sliders = len(self.slider_definitions)
        if total_sliders == 0:
            self.sliders = []
            self.slider_param_keys = []
            return

        max_columns = min(3, total_sliders)
        min_card_width = 180
        min_card_height = 52
        best_layout: tuple[int, int, float, float] | None = None
        fallback_layout: tuple[int, int, float, float] | None = None
        for columns in range(max_columns, 0, -1):
            width_available = panel.width - 2 * inner_padding - gap * (columns - 1)
            if width_available <= 0:
                continue
            card_width = width_available / columns
            if card_width < min_card_width:
                continue
            rows = math.ceil(total_sliders / columns)
            height_available = available_height - gap * (rows - 1)
            if height_available <= 0:
                continue
            card_height_raw = height_available / rows
            candidate = (columns, rows, card_width, card_height_raw)
            if card_height_raw >= min_card_height:
                best_layout = candidate
                break
            fallback_layout = candidate

        if best_layout is None:
            if fallback_layout is not None:
                columns, rows, card_width, card_height_raw = fallback_layout
            else:
                columns = 1
                rows = max(1, total_sliders)
                width_available = panel.width - 2 * inner_padding
                card_width = width_available if width_available > 0 else float(panel.width)
                height_available = available_height - gap * (rows - 1)
                if height_available <= 0:
                    height_available = available_height
                card_height_raw = height_available / rows if rows else float(available_height)
        else:
            columns, rows, card_width, card_height_raw = best_layout

        max_card_height = 120
        card_height = min(card_height_raw, max_card_height)

        self.sliders = []
        self.slider_param_keys = []
        for index, definition in enumerate(self.slider_definitions):
            row = index // columns
            col = index % columns
            x = panel.left + inner_padding + col * (card_width + gap)
            y = content_top + row * (card_height + gap)
            rect = (int(x), int(y), int(card_width), int(card_height))
            key = definition['key']
            current_value = self.demo_config['params'].get(key, self.initial_params[key])
            ratio = 0.0
            minimum, maximum = definition['bounds']
            if maximum > minimum:
                ratio = (current_value - minimum) / (maximum - minimum)
            slider = ParamSlider(
                self.app,
                definition['label'],
                rect,
                definition['bounds'],
                definition['step'],
                key,
                definition['decimals'],
                ratio,
                padding=12,
                label_size=18,
                value_size=20,
                track_height=6,
            )
            slider.set_value(current_value)
            self.sliders.append(slider)
            self.slider_param_keys.append(key)
            if key == 'T_left':
                self.slider_T_left = slider
            elif key == 'T_right':
                self.slider_T_right = slider
            elif key == 'accommodation':
                self.slider_accommodation = slider
            elif key == 'size_scale':
                self.slider_size_scale = slider
            elif key == 'tagged_count':
                self.slider_tag_count = slider
            elif key == 'slowmo':
                self.slider_slowmo = slider

    def _ensure_demo(self) -> None:
        assert self.sim_rect is not None
        params_initial = OrderedDict()
        for key, slider in zip(self.slider_param_keys, self.sliders):
            params_initial[key] = slider.getValue()
        if 'T' in self.demo_config['params']:
            params_initial['T'] = self.demo_config['params']['T']
        elif 'T' in self.initial_params:
            params_initial['T'] = self.initial_params['T']
        self.demo_config['params'] = OrderedDict(params_initial)
        if self.demo is None:
            self.demo = Demo(
                self.app,
                (self.sim_rect.left, self.sim_rect.top),
                (self.sim_rect.width, self.sim_rect.height),
                (255, 255, 255),
                self.border_color,
                self.bg_color,
                params_initial.copy(),
            )
            self.demo.set_dim_untracked(self.dim_active)
            self.demo.set_trail_enabled(self.trail_active)
        else:
            self.demo.resize_viewport((self.sim_rect.left, self.sim_rect.top), (self.sim_rect.width, self.sim_rect.height))
            self.demo.screen = self.app.screen

    # ------------------------------------------------------------------ Slider data
    def _round_value(self, value: float, decimals: int) -> float:
        return int(round(value, 0)) if decimals == 0 else round(value, decimals)

    def _build_slider_definitions(self):
        param_names, _, _, param_bounds, param_initial, param_step, par4sim, dec_numbers = self._load_params()
        original_bounds = list(param_bounds)

        entries = list(zip(param_names, param_bounds, param_initial, param_step, par4sim, dec_numbers))
        ignore_params = {'gamma', 'k', 'm_spring', 'R', 'R_spring', 'T'}
        base_initial_values: OrderedDict[str, float] = OrderedDict()
        base_decimals: dict[str, int] = {}
        slider_defs: list[dict] = []
        for name, bounds, initial_value, step, name_par, dec_number in entries:
            min_val, max_val = bounds
            if max_val <= min_val:
                clamped_value = min_val
                ratio = 0.0
            else:
                clamped_value = max(min_val, min(max_val, initial_value))
                ratio = (clamped_value - min_val) / (max_val - min_val)
            base_initial_values[name_par] = clamped_value
            base_decimals[name_par] = dec_number
            if name_par in ignore_params:
                continue
            slider_defs.append(
                {
                    'label': name,
                    'bounds': bounds,
                    'initial_pos': ratio,
                    'step': step,
                    'key': name_par,
                    'decimals': dec_number,
                }
            )

        # Additional sliders
        min_T = 100.0
        max_T = 1000.0
        initial_T_left = 600.0
        initial_T_right = 300.0
        min_accom = 0.0
        max_accom = 1.0
        initial_accom = 1.0
        size_bounds = (0.5, 1.5)
        size_initial = 1.3
        slowmo_bounds = (0.1, 1.0)
        slowmo_initial = 1.0
        tag_bounds = (1, 100)
        tag_initial = 10

        slider_defs.extend(
            [
                {
                    'label': self.lang['slider_T_left'],
                    'bounds': (min_T, max_T),
                    'initial_pos': (initial_T_left - min_T) / (max_T - min_T),
                    'step': (max_T - min_T) / 100.0,
                    'key': 'T_left',
                    'decimals': 0,
                },
                {
                    'label': self.lang['slider_T_right'],
                    'bounds': (min_T, max_T),
                    'initial_pos': (initial_T_right - min_T) / (max_T - min_T),
                    'step': (max_T - min_T) / 100.0,
                    'key': 'T_right',
                    'decimals': 0,
                },
                {
                    'label': self.lang['slider_accommodation'],
                    'bounds': (min_accom, max_accom),
                    'initial_pos': (initial_accom - min_accom) / (max_accom - min_accom),
                    'step': (max_accom - min_accom) / 100.0,
                    'key': 'accommodation',
                    'decimals': 2,
                },
                {
                    'label': self.lang['slider_size_scale'],
                    'bounds': size_bounds,
                    'initial_pos': (size_initial - size_bounds[0]) / (size_bounds[1] - size_bounds[0]),
                    'step': (size_bounds[1] - size_bounds[0]) / 100.0,
                    'key': 'size_scale',
                    'decimals': 2,
                },
                {
                    'label': self.lang['slider_tagged_count'],
                    'bounds': tag_bounds,
                    'initial_pos': (tag_initial - tag_bounds[0]) / (tag_bounds[1] - tag_bounds[0]),
                    'step': 1,
                    'key': 'tagged_count',
                    'decimals': 0,
                },
                {
                    'label': self.lang['slider_slowmo'],
                    'bounds': slowmo_bounds,
                    'initial_pos': (slowmo_initial - slowmo_bounds[0]) / (slowmo_bounds[1] - slowmo_bounds[0]),
                    'step': (slowmo_bounds[1] - slowmo_bounds[0]) / 100.0,
                    'key': 'slowmo',
                    'decimals': 2,
                },
            ]
        )

        initial_params = OrderedDict()
        for definition in slider_defs:
            min_val, max_val = definition['bounds']
            if max_val <= min_val:
                value = min_val
            else:
                value = min_val + definition['initial_pos'] * (max_val - min_val)
            initial_params[definition['key']] = self._round_value(value, definition['decimals'])

        if 'T' in base_initial_values:
            initial_params['T'] = self._round_value(base_initial_values['T'], base_decimals.get('T', 0))

        return slider_defs, original_bounds, initial_params

    # ------------------------------------------------------------------ Buttons actions
    def apply(self):
        self.demo.simulation.set_params(
            T_left=self.slider_T_left.getValue(),
            T_right=self.slider_T_right.getValue(),
            accommodation=self.slider_accommodation.getValue(),
        )
        self.demo_config['params']['T_left'] = self.slider_T_left.getValue()
        self.demo_config['params']['T_right'] = self.slider_T_right.getValue()
        self.demo_config['params']['accommodation'] = self.slider_accommodation.getValue()

        size_val = float(self.slider_size_scale.getValue())
        self.demo.update_radius_scale(size_val)
        self.demo_config['params']['size_scale'] = size_val

        self.tagged_count = int(self.slider_tag_count.getValue())
        self.demo_config['params']['tagged_count'] = self.tagged_count

        slowmo_val = float(self.slider_slowmo.getValue())
        self.demo.set_time_scale(slowmo_val)
        self.demo_config['params']['slowmo'] = slowmo_val

        self.demo._refresh_iter(self.demo_config)
        self.demo_config['is_changed'] = False

    def add_marked_particles(self):
        self.demo.add_tagged_particles(self.tagged_count)
        if 'r' in self.demo_config['params']:
            self.demo_config['params']['r'] += self.tagged_count
        self.demo_config['is_changed'] = True

    def toggle_dim_particles(self):
        self.dim_active = not self.dim_active
        self.demo.set_dim_untracked(self.dim_active)
        if self.dim_button is not None:
            new_label = self.dim_labels[1] if self.dim_active else self.dim_labels[0]
            self.dim_button._prep_msg(new_label)

    def toggle_trail(self):
        self.trail_active = not self.trail_active
        self.demo.set_trail_enabled(self.trail_active)
        if self.trail_button is not None:
            new_label = self.trail_labels[1] if self.trail_active else self.trail_labels[0]
            self.trail_button._prep_msg(new_label)

    def modes(self):
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
        param_step[1], param_step[2] = int(param_step[1]), int(param_step[2])
        par4sim = loader['par4sim']
        dec_numbers = [1, 0, 0, 0, 1, 0, 0]
        return param_names, sliders_gap, param_poses, param_bounds, param_initial, param_step, par4sim, dec_numbers

    # ------------------------------------------------------------------ Drawing
    def _update_screen(self):
        assert self.sim_rect is not None
        self.screen.blit(self.background, (0, 0))

        self._draw_simulation_shadow()
        self.demo.draw_check(self.demo_config)
        self._draw_simulation_border()
        self._draw_counters()

        self._draw_slider_panel_background()
        for slider in self.sliders:
            slider.draw_check(self.demo_config['params'])

        for button in self.buttons:
            button.draw_button()

        if self.right_panel_rect:
            self._draw_flux_panel()

    def _draw_simulation_shadow(self) -> None:
        rect = self.sim_rect
        shadow_rect = rect.copy()
        shadow_rect.x += 14
        shadow_rect.y += 18
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, self.shadow_color, shadow_surface.get_rect(), border_radius=24)
        self.screen.blit(shadow_surface, shadow_rect.topleft)

    def _draw_simulation_border(self) -> None:
        rect = self.sim_rect
        pygame.draw.rect(self.screen, self.border_color, rect, width=2, border_radius=20)

    def _draw_slider_panel_background(self) -> None:
        assert self.slider_panel_rect is not None
        rect = self.slider_panel_rect
        shadow_rect = rect.copy()
        shadow_rect.x += 12
        shadow_rect.y += 16
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, self.shadow_color, shadow_surface.get_rect(), border_radius=22)
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, self.panel_color, rect, border_radius=20)
        pygame.draw.rect(self.screen, self.border_color, rect, width=2, border_radius=20)

        title_surface = self.panel_title_font.render(self.panel_title_text, True, (38, 44, 60))
        title_rect = title_surface.get_rect()
        title_rect.topleft = (rect.left + max(24, rect.width // 20), rect.top + max(18, rect.height // 18))
        self.screen.blit(title_surface, title_rect)

    def _draw_flux_panel(self) -> None:
        assert self.right_panel_rect is not None
        rect = self.right_panel_rect
        shadow_rect = rect.copy()
        shadow_rect.x += 12
        shadow_rect.y += 16
        shadow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, self.shadow_color, shadow_surface.get_rect(), border_radius=22)
        self.screen.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(self.screen, self.panel_color, rect, border_radius=20)
        pygame.draw.rect(self.screen, self.border_color, rect, width=2, border_radius=20)

        title_surface = self.panel_title_font.render(self.graph_title_text, True, (38, 44, 60))
        title_rect = title_surface.get_rect()
        padding = max(24, rect.width // 20)
        title_rect.topleft = (rect.left + padding, rect.top + padding)
        self.screen.blit(title_surface, title_rect)

        last_flux = 0.0
        counts_text = None
        if self.demo and getattr(self.demo, 'simulation', None):
            last_flux = float(self.demo.simulation.get_last_midplane_flux())
            counts = self.demo.simulation.get_last_midplane_counts()
            forward = counts.get('left_to_right', 0)
            backward = counts.get('right_to_left', 0)
            template = self.graph_counts_template
            if '{forward' in template and '{backward' in template:
                counts_text = template.format(forward=forward, backward=backward)
            else:
                counts_text = f'-> {forward}   <- {backward}'
        value_text = None
        template = self.graph_value_template
        if '{value' in template:
            value_text = template.format(value=last_flux)
        else:
            value_text = f'{template} {last_flux:+.1f}'

        text_y = title_rect.bottom + max(10, padding // 4)
        if value_text:
            value_surface = self.graph_value_font.render(value_text, True, (42, 48, 66))
            value_rect = value_surface.get_rect()
            value_rect.topleft = (rect.left + padding, text_y)
            self.screen.blit(value_surface, value_rect)
            text_y = value_rect.bottom + max(6, padding // 5)
        if counts_text:
            counts_surface = self.graph_counts_font.render(counts_text, True, (110, 118, 134))
            counts_rect = counts_surface.get_rect()
            counts_rect.topleft = (rect.left + padding, text_y)
            self.screen.blit(counts_surface, counts_rect)
            text_y = counts_rect.bottom + max(12, padding // 3)
        else:
            text_y += max(12, padding // 3)

        graph_bottom = rect.bottom - padding
        if graph_bottom <= text_y + 20:
            return

        graph_rect = pygame.Rect(rect.left + padding, text_y, rect.width - 2 * padding, graph_bottom - text_y)
        pygame.draw.rect(self.screen, (254, 255, 255), graph_rect, border_radius=18)
        pygame.draw.rect(self.screen, self.border_color, graph_rect, width=1, border_radius=18)

        inner_margin = max(12, int(graph_rect.width * 0.04))
        inner_margin = min(inner_margin, graph_rect.width // 2 - 1 if graph_rect.width > 2 else inner_margin)
        inner_margin = min(inner_margin, graph_rect.height // 2 - 1 if graph_rect.height > 2 else inner_margin)
        plot_rect = graph_rect.inflate(-2 * inner_margin, -2 * inner_margin)
        if plot_rect.width <= 2 or plot_rect.height <= 2:
            plot_rect = graph_rect.inflate(-8, -8)

        samples: list[tuple[float, float]] = []
        if self.demo:
            samples = list(self.demo.midplane_flux_samples)
        if len(samples) > self.graph_max_samples:
            samples = samples[-self.graph_max_samples:]

        if self.demo and len(samples) >= 2 and plot_rect.width > 1 and plot_rect.height > 1:
            self.demo.draw_midplane_flux_graph(
                self.screen,
                plot_rect,
                samples=samples,
                line_color=self.primary_color,
                baseline_color=(186, 190, 208),
                background=(0, 0, 0, 0),
            )
        else:
            placeholder_surface = self.placeholder_font.render(self.placeholder_text, True, (120, 128, 146))
            placeholder_rect = placeholder_surface.get_rect()
            placeholder_rect.center = plot_rect.center if plot_rect.width > 1 and plot_rect.height > 1 else graph_rect.center
            self.screen.blit(placeholder_surface, placeholder_rect)

    def _draw_counters(self):
        """Render counters showing how many tagged particles hit each wall."""
        if not self.demo or not self.demo.has_tagged_particles():
            return
        left_hits, right_hits = self.demo.get_wall_hit_counts()
        left_text = self.counter_left_template.format(count=left_hits)
        right_text = self.counter_right_template.format(count=right_hits)
        left_surface = self.counter_font.render(left_text, True, (220, 70, 40))
        right_surface = self.counter_font.render(right_text, True, (60, 130, 255))
        padding = 20
        top_y = self.sim_rect.top + padding
        left_x = self.sim_rect.left + padding
        right_x = self.sim_rect.left + self.sim_rect.width - right_surface.get_width() - padding
        self.screen.blit(left_surface, (left_x, top_y))
        self.screen.blit(right_surface, (right_x, top_y))

    # ------------------------------------------------------------------ Events
    def _check_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                quit()
            elif event.type == pygame.VIDEORESIZE:
                self.app.handle_resize(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = pygame.mouse.get_pos()
                self._check_buttons(mouse_position)
        mouse_pos = pygame.mouse.get_pos()
        mouse = pygame.mouse.get_pressed()
        self._check_sliders(mouse_pos, mouse)

    def _check_buttons(self, mouse_position):
        for button in self.buttons:
            if button.rect.collidepoint(mouse_position):
                button.command()

    def _check_sliders(self, mouse_position, mouse_pressed):
        for slider in self.sliders:
            slider.slider.hovered = False
            if slider.slider.button_rect.collidepoint(mouse_position):
                if mouse_pressed[0] and not self.slider_grabbed:
                    slider.slider.grabbed = True
                    self.slider_grabbed = True
            if not mouse_pressed[0]:
                slider.slider.grabbed = False
                self.slider_grabbed = False
            if slider.slider.button_rect.collidepoint(mouse_position):
                slider.slider.hovered = True
            if slider.slider.grabbed:
                slider.slider.move_slider(mouse_position)
                slider.slider.hovered = True

    def correct_limits(self):
        return

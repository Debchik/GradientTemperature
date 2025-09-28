from button import Button
import pygame


class ParamSlider:
    def __init__(self, app, name, position, bounds, step, name_par, dec_number,
                 initial_pos,
                 **kwargs):
        self.name_par = name_par
        self.dec_round = lambda x: int(round(x, 0)) if dec_number == 0 else round(x, dec_number)
        self.init_values = {
            'slider': [250, 15],
            'sl_value': {
                'offset': [200, 9],
                'size': (30, 25)
            },
            'par_name': {
                'offset': [430, 14],
                'size': (50, 30)
            }
        }

        self.slider = _SliderImpl(app, position, self.init_values['slider'], initial_pos, bounds[0], bounds[1])

        self.sl_val = Button(app, str(self.slider.getValue()),
                             list((position[i] - self.init_values['sl_value']['offset'][i] for i in range(2))),
                             self.init_values['sl_value']['size'], **kwargs)
        self.par_name = Button(app, name,
                               list((position[i] - self.init_values['par_name']['offset'][i] for i in range(2))),
                               self.init_values['par_name']['size'], **kwargs)

    def draw_check(self, params):
        self.slider.draw()
        val = self.dec_round(self.slider.getValue())
        params[self.name_par] = val
        self.sl_val._prep_msg(str(val))
        self.sl_val.draw_button()
        self.par_name.draw_button()

    def getValue(self):
        return self.dec_round(self.slider.getValue())


BUTTONSTATES = {True: 'white', False: (100, 100, 100)}


class _SliderImpl:
    def __init__(self, app, pos: tuple, size: tuple, initial_val: float, min: int, max: int) -> None:
        self.pos = pos
        self.size = size
        self.hovered = False
        self.grabbed = False
        self.screen = app.screen

        self.slider_left_pos = self.pos[0] - (size[0] // 2)
        self.slider_right_pos = self.pos[0] + (size[0] // 2)
        self.slider_top_pos = self.pos[1] - (size[1] // 2)

        self.min = min
        self.max = max
        self.initial_val = (self.slider_right_pos - self.slider_left_pos) * initial_val  # <- percentage

        self.container_rect = pygame.Rect(self.slider_left_pos, self.slider_top_pos, self.size[0], self.size[1])
        self.button_rect = pygame.Rect(self.slider_left_pos + self.initial_val - 10, self.slider_top_pos, 20,
                                       self.size[1])

    def move_slider(self, mouse_pos):
        pos = mouse_pos[0]
        if pos < self.slider_left_pos:
            pos = self.slider_left_pos
        if pos > self.slider_right_pos:
            pos = self.slider_right_pos
        self.button_rect.centerx = pos

    def hover(self):
        self.hovered = True

    def draw(self):
        pygame.draw.rect(self.screen, "darkgray", self.container_rect)
        pygame.draw.rect(self.screen, BUTTONSTATES[self.hovered], self.button_rect)

    def getValue(self):
        val_range = self.slider_right_pos - self.slider_left_pos - 1
        button_val = self.button_rect.centerx - self.slider_left_pos

        return (button_val / val_range) * (self.max - self.min) + self.min

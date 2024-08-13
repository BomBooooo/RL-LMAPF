import drawsvg as dw
import numpy as np


class Render:
    def __init__(self, w, h, cell_size, colors, interval=1, seed=None):
        self.draw_size = (w, h)
        self._interval = interval
        self._grid_size = (w, h)
        self._cell = cell_size
        self._half_cell = cell_size / 2
        self._cell_div5 = cell_size / 5
        self._kiva_size = 0.95 * self._cell
        self._adapt_pos = 0.475 * self._cell
        self._shelf_color, self._o_color = colors
        self.rng = np.random.default_rng(seed)

    def add_rectangle(self, x, y, color):
        self.draw.append(
            dw.Rectangle(
                x * self._cell,
                y * self._cell,
                # (x - 0.5) * self._cell,
                # (y - 0.5) * self._cell,
                self._cell,
                self._cell,
                fill=color,
                stroke=color,
            )
        )

    def _add_edge(self):
        for i in range(self.draw_size[0]):
            self.add_rectangle(i, -1, self._o_color)
            self.add_rectangle(i, self.draw_size[1], self._o_color)
        for j in range(self.draw_size[1]):
            self.add_rectangle(-1, j, self._o_color)
            self.add_rectangle(self.draw_size[0], j, self._o_color)

    def _add_items(self, pos, color):
        for p in pos:
            self.add_rectangle(p[0], p[1], color)

    def init_draw(self, pos, goal, obstacle_set, shelves_set, duration=500):
        self.agent_num = pos.shape[0]
        self.colors = self.rng.random((self.agent_num, 3)) * 255
        self.draw = dw.Drawing(
            self.draw_size[0] * self._cell,
            self.draw_size[1] * self._cell,
            # Seconds
            animation_config=dw.types.SyncedAnimationConfig(
                duration=duration * self._interval
            ),
        )
        self.agents_draw = []
        self.goal_draw = []
        for i in range(self.agent_num):
            c = f"rgb({self.colors[i, 0]},{self.colors[i, 1]},{self.colors[i, 2]})"
            a = dw.Use(self.g_Kiva(i, color=c), 0, 0)
            a.add_key_frame(
                0, x=pos[i, 0].item() * self._cell, y=pos[i, 1].item() * self._cell
            )
            self.agents_draw.append(a)
            g = dw.Use(
                dw.Circle(
                    0,
                    0,
                    0.35 * self._cell,
                    fill="white",
                    fill_opacity=0,
                    stroke=c,
                    stroke_width=self._cell * 0.1,
                ),
                0,
                0,
            )
            g.add_key_frame(
                0,
                x=goal[i, 0].item() * self._cell + self._half_cell,
                y=goal[i, 1].item() * self._cell + self._half_cell,
            )
            self.goal_draw.append(g)

        self._add_edge()
        self._add_items(obstacle_set, self._o_color)
        self._add_items(shelves_set, self._shelf_color)

    def update(self, t, pos, goal):
        time_step = t * self._interval
        for i in range(self.agent_num):
            self.agents_draw[i].add_key_frame(
                time_step,
                x=pos[i, 0].item() * self._cell,
                y=pos[i, 1].item() * self._cell,
            )
            self.goal_draw[i].add_key_frame(
                time_step,
                x=goal[i, 0].item() * self._cell + self._half_cell,
                y=goal[i, 1].item() * self._cell + self._half_cell,
            )

    def save(self, path, duration):
        for a in self.agents_draw:
            self.draw.append(a)
        for g in self.goal_draw:
            self.draw.append(g)
        time_step = []
        text = []
        for i in range(duration):
            time_step.append(i)
            text.append(str(i))
        dw.native_animation.animate_text_sequence(
            self.draw,
            time_step,
            text,
            self._cell,
            (self._grid_size[0] / 2) * self._cell,
            self._cell,
            fill="white",
        )
        self.agents_draw.clear()
        self.goal_draw.clear()
        self.draw.save_svg(path)

    def g_Kiva(self, name, color):
        g_Kiva = dw.Group(stroke="black")
        g_Kiva.append(
            dw.Rectangle(
                0,
                0,
                self._kiva_size,
                self._kiva_size,
                ry=f"{self._cell_div5}",
                fill=color,
            )
        )
        g_Kiva.append(
            dw.Circle(self._adapt_pos, self._adapt_pos, 0.3 * self._cell, fill="white")
        )
        g_Kiva.append(
            dw.Text(
                str(name),
                self._cell_div5,
                self._adapt_pos,
                self._adapt_pos + (self._cell_div5 / 3),
                text_anchor="middle",
            )
        )
        return g_Kiva

    def g_shelf(self, name):
        shelf_size = 0.6 * self._cell
        shelf_pos = (self._kiva_size - shelf_size) / 2
        g_shelf = dw.Group(stroke="black")
        g_shelf.append(
            dw.Rectangle(
                shelf_pos,
                shelf_pos,
                shelf_size,
                shelf_size,
                fill=self._shelf_color,
            )
        )
        g_shelf.append(
            dw.Text(
                str(name),
                self._cell_div5,
                self._adapt_pos,
                self._adapt_pos + (self._cell_div5 / 3),
                text_anchor="middle",
            )
        )
        return g_shelf

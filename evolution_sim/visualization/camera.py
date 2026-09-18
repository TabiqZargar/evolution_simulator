"""Camera: world ↔ screen coordinate transforms and pan/zoom state."""

from __future__ import annotations

import pygame


class Camera:
    def __init__(self, world_w: int, world_h: int, screen_w: int, screen_h: int) -> None:
        self.world_w = world_w
        self.world_h = world_h
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._zoom = min(screen_w / max(world_w, 1), screen_h / max(world_h, 1)) * 0.9
        self._cx = world_w / 2.0
        self._cy = world_h / 2.0
        self._recompute()

    @property
    def zoom(self) -> float:
        return self._zoom

    def _recompute(self) -> None:
        self._offset_x = int(self.screen_w / 2 - self._cx * self._zoom)
        self._offset_y = int(self.screen_h / 2 - self._cy * self._zoom)

    def resize(self, screen_w: int, screen_h: int) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._recompute()

    # Transformations -------------------------------------------------------
    def world_to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        sx = int(wx * self._zoom + self._offset_x)
        sy = int(wy * self._zoom + self._offset_y)
        return sx, sy

    def screen_to_world(self, sx: int, sy: int) -> tuple[float, float]:
        wx = (sx - self._offset_x) / max(self._zoom, 0.001)
        wy = (sy - self._offset_y) / max(self._zoom, 0.001)
        return wx, wy

    def radius_on_screen(self, r_world: float) -> int:
        return max(1, int(r_world * self._zoom))

    # Interaction -----------------------------------------------------------
    def pan(self, dx_screen: int, dy_screen: int) -> None:
        self._cx -= dx_screen / max(self._zoom, 0.001)
        self._cy -= dy_screen / max(self._zoom, 0.001)
        self._clamp()
        self._recompute()

    def zoom_at(self, mx: int, my: int, factor: float) -> None:
        wx, wy = self.screen_to_world(mx, my)
        self._zoom *= factor
        self._zoom = max(self._zoom, 1.5)
        self._zoom = min(self._zoom, 250.0)
        self._cx = wx
        self._cy = wy
        self._clamp()
        self._recompute()

    def _clamp(self) -> None:
        margin = 5.0
        self._cx = max(min(self._cx, self.world_w + margin), -margin)
        self._cy = max(min(self._cy, self.world_h + margin), -margin)

    # Visible world rect (integer px bounds) --------------------------------
    def visible_rect(self) -> pygame.Rect:
        x0, y0 = self.screen_to_world(0, 0)
        x1, y1 = self.screen_to_world(self.screen_w, self.screen_h)
        margin = 2.0
        return pygame.Rect(
            int(x0 - margin),
            int(y0 - margin),
            int(x1 - x0 + margin * 2),
            int(y1 - y0 + margin * 2),
        )

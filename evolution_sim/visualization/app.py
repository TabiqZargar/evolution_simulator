"""Pygame application loop for the evolution simulator."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

import pygame

from evolution_sim.config import SimulationConfig
from evolution_sim.persistence import load_state, save_state
from evolution_sim.simulation.engine import Engine
from evolution_sim.simulation.organism import Organism
from evolution_sim.visualization.camera import Camera
from evolution_sim.visualization.colors import PANEL_BG, PANEL_BORDER, TEXT_DIM
from evolution_sim.visualization.renderer import Renderer
from evolution_sim.visualization.ui import (
    draw_ancestry_panel,
    draw_charts,
    draw_info_panel,
    draw_selection_panel,
    draw_text,
)


class Application:
    def __init__(self, engine: Engine, args: argparse.Namespace) -> None:
        self.engine = engine
        self.args = args
        self.screen_w = 1280
        self.screen_h = 800
        pygame.init()
        pygame.display.set_caption("Evolution Simulator")
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.camera = Camera(self.engine.world.width, self.engine.world.height, self.screen_w, self.screen_h)
        self.renderer = Renderer(self.engine)

        self.font = pygame.font.Font(None, 18)
        self.small = pygame.font.Font(None, 15)
        self.header_font = pygame.font.Font(None, 22)

        self.running = False
        self.paused = False
        self.simulating = True
        self.speed = 2
        self.selected: Optional[Organism] = None
        self._dragging = False
        self._last_mouse = (0, 0)

    # ------------------------------------------------------------------ input
    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self.screen_w, self.screen_h = event.w, event.h
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.RESIZABLE)
            self.camera.resize(self.screen_w, self.screen_h)
        elif event.type == pygame.KEYDOWN:
            self._on_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if event.button == 1:
                self._dragging = True
                self._last_mouse = (mx, my)
                wx, wy = self.camera.screen_to_world(mx, my)
                self.selected = self._pick_organism(wx, wy)
            elif event.button == 4:
                self.camera.zoom_at(mx, my, 1.15)
            elif event.button == 5:
                self.camera.zoom_at(mx, my, 1 / 1.15)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self._dragging:
                mx, my = event.pos
                dx = mx - self._last_mouse[0]
                dy = my - self._last_mouse[1]
                self.camera.pan(dx, dy)
                self._last_mouse = (mx, my)

    def _on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
            self.simulating = not self.simulating and not self.paused
        elif key == pygame.K_n:
            self.paused = True
            self.simulating = False
            if self.engine.alive_count():
                self.engine.step()
        elif key == pygame.K_g:
            self.paused = True
            self.simulating = False
            if self.engine.alive_count():
                self.engine.advance_generation()
        elif key == pygame.K_t:
            modes = self.renderer.TRAIT_MODES
            self.renderer.trait_mode = modes[(modes.index(self.renderer.trait_mode) + 1) % len(modes)]
        elif key == pygame.K_UP or key == pygame.K_EQUALS:
            self.speed = min(self.speed + 1, 50)
        elif key == pygame.K_DOWN or key == pygame.K_MINUS:
            self.speed = max(self.speed - 1, 1)
        elif key == pygame.K_s:
            os.makedirs("saves", exist_ok=True)
            save_state(self.engine, "saves/save.json")
            sys.stderr.write("saved saves/save.json\n")
        elif key == pygame.K_l:
            if os.path.exists("saves/save.json"):
                self.engine = load_state("saves/save.json")
                self.renderer = Renderer(self.engine)
                self.camera = Camera(
                    self.engine.world.width, self.engine.world.height, self.screen_w, self.screen_h
                )
                sys.stderr.write("loaded saves/save.json\n")

    def _pick_organism(self, wx: float, wy: float) -> Optional[Organism]:
        best = None
        best_d = 1e9
        radius = 10.0 / max(self.camera.zoom, 0.001)
        for org in self.engine.world.organisms:
            d2 = (org.x - wx) ** 2 + (org.y - wy) ** 2
            if d2 <= radius * radius and d2 < best_d:
                best_d = d2
                best = org
        return best

    # ------------------------------------------------------------------- loop
    def run(self) -> int:
        self.running = True
        self.simulating = not self.paused
        while self.running:
            for event in pygame.event.get():
                self._handle_event(event)

            if self.simulating and not self.paused:
                for _ in range(self.speed):
                    if not self.engine.alive_count():
                        break
                    self.engine.step()

            self._draw_frame()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        return 0

    # ------------------------------------------------------------------ draw
    def _draw_frame(self) -> None:
        self.renderer.draw(self.screen, self.camera, self.selected)
        self._draw_hud()
        self._draw_hints()

    def _draw_hud(self) -> None:
        info_rect = draw_info_panel(
            self.screen, self.font, self.header_font, self.engine, self.simulating
        )
        draw_selection_panel(self.screen, self.font, self.header_font, self.engine, self.selected)
        draw_ancestry_panel(
            self.screen,
            self.font,
            self.header_font,
            self.engine,
            self.selected,
            self.engine.config.ancestry_depth,
        )

        chart_rect = pygame.Rect(12, info_rect.bottom + 8, 340, 190)
        pygame.draw.rect(self.screen, PANEL_BG, chart_rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, chart_rect, width=1, border_radius=6)
        draw_text(self.screen, self.header_font, "History", chart_rect.x + 10, chart_rect.y + 6)
        draw_charts(self.screen, chart_rect, self.small, self.engine.history)

        status = "RUNNING" if self.simulating else ("PAUSED" if self.paused else "STEPPING")
        draw_text(
            self.screen,
            self.small,
            status + f"   speed x{self.speed}   coloring: {self.renderer.trait_mode}",
            12,
            chart_rect.bottom + 6,
            TEXT_DIM,
        )

    def _draw_hints(self) -> None:
        hints = (
            "SPACE pause  N step  G generation  T color  +/- speed  "
            "S save  L load  click select  drag pan  wheel zoom  ESC quit"
        )
        draw_text(self.screen, self.small, hints, 12, self.screen_h - 22, TEXT_DIM, shadow=False)


def run(
    config: SimulationConfig,
    args: argparse.Namespace,
    overrides: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> int:
    if args is None:
        args = argparse.Namespace()
    load_path = getattr(args, "load", None)
    if load_path:
        engine = load_state(load_path)
        if overrides:
            engine.config = config
        print(f"resumed from {load_path} at tick {engine.tick}", file=sys.stderr)
    else:
        engine = Engine(config)
        engine.init()
    return Application(engine, args).run()

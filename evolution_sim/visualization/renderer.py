"""Rendering: terrain surface + food + organisms + event banner."""

from __future__ import annotations

import pygame

from evolution_sim.simulation.engine import Engine
from evolution_sim.simulation.organism import Organism
from evolution_sim.visualization.camera import Camera
from evolution_sim.visualization.colors import (
    BIOME_COLORS,
    FOOD_COLOR,
    FOOD_DEPLETED_COLOR,
    species_color,
    trait_color,
)

_WORLD_CELL_SCALE = 2  # terrain surface supersampling factor


class Renderer:
    """Draws the world onto a pygame screen every frame."""

    TRAIT_MODES = (
        "speed",
        "vision",
        "size",
        "metabolism",
        "efficiency",
        "temperature_tolerance",
        "aggression",
        "species",
    )

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.config = engine.config
        self.world = engine.world
        self.trait_mode = "species"
        self._terrain: pygame.Surface | None = None

    # Terrain ----------------------------------------------------------------
    def _ensure_terrain(self) -> pygame.Surface:
        if self._terrain is not None:
            return self._terrain
        terrain = self.world.terrain
        w = terrain.width * _WORLD_CELL_SCALE
        h = terrain.height * _WORLD_CELL_SCALE
        surface = pygame.Surface((w, h))
        for cy in range(terrain.height):
            for cx in range(terrain.width):
                biome = terrain.biome_at(cx, cy)
                base = BIOME_COLORS[biome]
                fert = terrain.fertility_at(cx, cy)
                shade = 0.75 + 0.45 * fert
                color = (
                    min(255, int(base[0] * shade)),
                    min(255, int(base[1] * shade)),
                    min(255, int(base[2] * shade)),
                )
                cxs = cx * _WORLD_CELL_SCALE
                cys = cy * _WORLD_CELL_SCALE
                surface.fill(color, (cxs, cys, _WORLD_CELL_SCALE, _WORLD_CELL_SCALE))
        self._terrain = surface
        return surface

    # Draw -------------------------------------------------------------------
    def draw(self, screen: pygame.Surface, camera: Camera, selected: Organism | None) -> None:
        screen.fill((4, 6, 9))
        self._draw_terrain(screen, camera)
        self._draw_food(screen, camera)
        self._draw_organisms(screen, camera)
        if selected is not None:
            self._draw_selected(screen, camera, selected)

    def _draw_terrain(self, screen: pygame.Surface, camera: Camera) -> None:
        terrain_surf = self._ensure_terrain()
        world_px_w = self.world.width * _WORLD_CELL_SCALE
        world_px_h = self.world.height * _WORLD_CELL_SCALE
        # stretch the supersampled terrain onto the projected screen rectangle
        target_w = max(1, int(world_px_w * camera.zoom / _WORLD_CELL_SCALE))
        target_h = max(1, int(world_px_h * camera.zoom / _WORLD_CELL_SCALE))
        scaled = pygame.transform.smoothscale(terrain_surf, (target_w, target_h))
        screen.blit(scaled, (int(camera._offset_x), int(camera._offset_y)))

    def _draw_food(self, screen: pygame.Surface, camera: Camera) -> None:
        view = camera.visible_rect()
        for patch in self.world.patches:
            if not view.collidepoint(patch.x, patch.y):
                continue
            sx, sy = camera.world_to_screen(patch.x, patch.y)
            if patch.quantity <= 0.0:
                continue
            frac = patch.quantity / max(patch.max_quantity, 1e-9)
            radius = max(1, int(1.0 * camera.zoom * (0.45 * frac + 0.2)))
            if radius >= 2:
                alpha_color = FOOD_COLOR + (
                    90 + int(120 * frac),
                )
                pygame.draw.circle(screen, alpha_color, (sx, sy), radius)
            else:
                screen.set_at((sx, sy), FOOD_DEPLETED_COLOR)

    def _organism_color(self, org: Organism) -> tuple[int, int, int]:
        if org.is_predator:
            a = org.trait("attack")
            return (220, 70, 44 + int(90 * a))
        if self.trait_mode == "species":
            return species_color(org.species_id)
        return trait_color(org.trait(self.trait_mode))

    def _draw_organisms(self, screen: pygame.Surface, camera: Camera) -> None:
        view = camera.visible_rect()
        for org in self.world.organisms:
            if not org.alive:
                continue
            if not view.collidepoint(org.x, org.y):
                continue
            sx, sy = camera.world_to_screen(org.x, org.y)
            radius = max(2, int((1.2 + 1.3 * org.size) * camera.zoom))
            color = self._organism_color(org)
            pygame.draw.circle(screen, color, (sx, sy), radius)
            pygame.draw.circle(screen, (0, 0, 0), (sx, sy), radius, 1)
            if not org.is_predator and org.energy < self.config.rest_energy_threshold:
                pygame.draw.circle(screen, (200, 40, 40), (sx, sy), max(1, radius - 2))

    def _draw_selected(self, screen: pygame.Surface, camera: Camera, org: Organism) -> None:
        sx, sy = camera.world_to_screen(org.x, org.y)
        radius = max(4, int((2.0 + 1.5 * org.size) * camera.zoom))
        ring = (255, 255, 255) if org.alive else (140, 140, 150)
        pygame.draw.circle(screen, ring, (sx, sy), radius, 2)
        for parent_id in (org.parent_a, org.parent_b):
            if parent_id is None:
                continue
            parent = self.engine.organism_by_id(parent_id)
            if parent is None:
                continue
            px, py = camera.world_to_screen(parent.x, parent.y)
            pygame.draw.line(screen, (190, 190, 60), (sx, sy), (px, py), 1)

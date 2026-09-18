"""Lightweight chart and text helpers drawn on the pygame screen."""

from __future__ import annotations

import pygame

from evolution_sim.analytics.history import History
from evolution_sim.simulation.engine import Engine
from evolution_sim.simulation.organism import Organism
from evolution_sim.visualization.colors import (
    ACCENT,
    GOOD,
    PANEL_BG,
    PANEL_BORDER,
    TEXT,
    TEXT_DIM,
    WARN,
)


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = TEXT,
    shadow: bool = True,
) -> None:
    if shadow:
        surface.blit(font.render(text, True, (0, 0, 0)), (x + 1, y + 1))
    surface.blit(font.render(text, True, color), (x, y))


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, PANEL_BG, rect, border_radius=6)
    pygame.draw.rect(surface, PANEL_BORDER, rect, width=1, border_radius=6)
    if title:
        draw_text(surface, font, title, rect.x + 10, rect.y + 8, ACCENT)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
class Chart:
    """A tiny labelled line chart drawn with raw rects/lines."""

    def __init__(self, series: list[float], color: tuple[int, int, int], label: str) -> None:
        self.series = series
        self.color = color
        self.label = label


def draw_charts(
    surface: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    history: History,
) -> None:
    if not history.entries:
        return
    charts = [
        Chart(history.series("avg_fitness"), GOOD, "fitness"),
        Chart(history.series("diversity"), ACCENT, "diversity"),
        Chart(history.series("population"), TEXT_DIM, "population"),
        Chart(history.series("avg_speed"), WARN, "speed"),
    ]
    plot_w = rect.width - 16
    plot_h = rect.height - 30
    plots_per_row = 2
    per_plot_w = plot_w // plots_per_row
    per_plot_h = plot_h // 2

    for idx, chart in enumerate(charts):
        col = idx % plots_per_row
        row = idx // plots_per_row
        px = rect.x + 10 + col * (per_plot_w + 6)
        py = rect.y + 26 + row * (per_plot_h + 6)
        box = pygame.Rect(px, py, per_plot_w - 6, per_plot_h - 6)
        pygame.draw.rect(surface, (10, 13, 18), box, border_radius=4)
        draw_text(surface, font, chart.label, box.x + 4, box.y + 2, chart.color)
        pts: list[tuple[int, int]] = []
        series = chart.series
        lo = min(series) if series else 0.0
        hi = max(series) if series else 1.0
        span = max(hi - lo, 1.0)
        n = max(len(series), 1)
        inner_h = max(box.h - 18, 4)
        for i, value in enumerate(series):
            sx = box.x + 4 + int((box.w - 8) * i / max(n - 1, 1))
            sy = box.bottom - 6 - int(inner_h * (value - lo) / span)
            pts.append((sx, sy))
        if len(pts) >= 2:
            pygame.draw.lines(surface, chart.color, False, pts, 2)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def draw_info_panel(
    surface: pygame.Surface,
    font: pygame.font.Font,
    header_font: pygame.font.Font,
    engine: Engine,
    simulating: bool,
) -> pygame.Rect:
    cfg = engine.config
    stats = engine.history.last()
    temp = engine.current_temperature()
    label = "frozen/scorching" if temp <= 0.2 or temp >= 0.8 else ""

    prey = engine.alive_prey_count() if cfg.predators_enabled else engine.alive_count()
    pred = len(engine.alive_predators()) if cfg.predators_enabled else 0

    lines = [
        f"Generation {engine.generation}   tick {engine.tick}",
        f"prey {prey}  predators {pred}",
        f"temperature {temp:.2f}  {label}".rstrip(),
        f"food {engine.world.total_food():.0f}",
    ]
    if stats is not None:
        lines.append(
            f"fit {stats.avg_fitness:.3f} (max {stats.max_fitness:.3f})  "
            f"div {stats.diversity:.3f}  species {stats.species_count}"
        )
    event = engine.event_manager.strongest_event()
    if event is not None:
        lines.append(f"{event.label()}  mag {event.magnitude:.2f}")

    pad = 10
    width = max(font.size(line)[0] for line in lines) + pad * 2
    height = len(lines) * 18 + pad * 2
    rect = pygame.Rect(12, 12, width, height)
    draw_panel(surface, rect, "Evolution", header_font)
    for i, line in enumerate(lines):
        draw_text(surface, font, line, rect.x + pad, rect.y + 30 + i * 18)
    return rect


def draw_selection_panel(
    surface: pygame.Surface,
    font: pygame.font.Font,
    header_font: pygame.font.Font,
    engine: Engine,
    org: Organism | None,
) -> None:
    if org is None:
        return
    lines = [
        f"id {org.organism_id}   {'ALIVE' if org.alive else 'dead: ' + (org.death_cause or '?')}",
        f"species {org.species_id}   gen {org.generation}",
        f"age {org.age}   energy {org.energy:.1f}   health {org.health:.2f}",
        f"children {org.children}   fitness {org.final_fitness if org.final_fitness is not None else '—'}",
        "",
    ]
    for trait in ("speed", "vision", "size", "metabolism", "efficiency", "fertility", "lifespan", "aggression"):
        lines.append(f"{trait:<22} {org.trait(trait):.3f}")
    if org.is_predator:
        lines.append(f"{'attack':<22} {org.attack:.3f}")

    pad = 10
    width = max(font.size(ln)[0] for ln in lines) + pad * 2
    height = len(lines) * 16 + pad * 2
    rect = pygame.Rect(surface.get_width() - width - 12, 12, width, height)
    draw_panel(surface, rect, "Selected organism", header_font)
    for i, line in enumerate(lines):
        draw_text(surface, font, line, rect.x + pad, rect.y + 30 + i * 16)


def draw_ancestry_panel(
    surface: pygame.Surface,
    font: pygame.font.Font,
    header_font: pygame.font.Font,
    engine: Engine,
    org: Organism | None,
    max_depth: int,
) -> None:
    if org is None:
        return
    rows: list[tuple[int, str]] = []
    current: Organism | None = org
    seen: set[int] = set()
    depth = 0
    while current is not None and depth < max_depth and current.organism_id not in seen:
        seen.add(current.organism_id)
        rows.append((current.organism_id, f"id {current.organism_id}"))
        nxt = None
        for pid in (current.parent_a, current.parent_b):
            candidate = engine.organism_by_id(pid) if pid is not None else None
            if candidate is not None:
                nxt = candidate
                break
        current = nxt
        depth += 1

    lines = [f"  id {org.organism_id} (you)"]
    for pid, _ in rows[1:]:
        lines.append(f"↑ id {pid}")
    label = "Ancestry" + (f" (clipped at {max_depth})" if len(rows) == max_depth else "")
    pad = 10
    width = max(font.size(ln)[0] for ln in lines) + pad * 2
    height = len(lines) * 16 + pad * 2
    rect = pygame.Rect(surface.get_width() - width - 12, 150, width, height)
    draw_panel(surface, rect, label, header_font)
    for i, line in enumerate(lines):
        draw_text(surface, font, line, rect.x + pad, rect.y + 30 + i * 16)

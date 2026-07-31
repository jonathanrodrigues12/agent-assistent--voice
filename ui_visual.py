"""
Assistant visual panel: a window with an animated circle.
Idle = gentle breathing. Listening = blue pulse. Speaking = colored waves
radiating out, inspired by a circular waveform.
Runs in a separate thread, controlled by VisualPanel.set_state().
"""

import math
import threading
import time

import pygame

WIDTH, HEIGHT = 420, 420
CENTER = (WIDTH // 2, HEIGHT // 2)
BASE_RADIUS = 110

BACKGROUND_COLOR = (12, 12, 20)
RING_COLOR = (20, 20, 35)
RING_BORDER_COLOR = (235, 235, 245)

WAVE_COLORS = [
    (255, 111, 97),   # coral
    (79, 168, 216),   # blue
    (124, 92, 191),   # purple
]
LISTENING_COLOR = (79, 168, 216)
LOADING_COLOR = (200, 200, 210)

NUM_RAYS = 64
NUM_LOADING_DOTS = 8


class VisualPanel:
    def __init__(self, title="Assistente", agent_name=None):
        self._state = "dormindo"  # dormindo | ouvindo | pensando | falando
        self._agent_name = agent_name
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._title = title
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def set_state(self, state):
        with self._lock:
            self._state = state

    def set_agent_name(self, agent_name):
        with self._lock:
            self._agent_name = agent_name

    def _read_state(self):
        with self._lock:
            return self._state

    def _read_agent_name(self):
        with self._lock:
            return self._agent_name

    def stop(self):
        self._running.clear()

    def _loop(self):
        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(self._title)
        clock = pygame.time.Clock()
        agent_name_font = pygame.font.SysFont(None, 16)

        t = 0.0
        phases = [i * (2 * math.pi / NUM_RAYS) for i in range(NUM_RAYS)]

        while self._running.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running.clear()

            state = self._read_state()
            screen.fill(BACKGROUND_COLOR)

            if state == "falando":
                self._draw_waves(screen, t, phases, intensity=1.0)
                ring_radius = BASE_RADIUS + 6 * math.sin(t * 6)
            elif state == "ouvindo":
                self._draw_waves(screen, t, phases, intensity=0.35, single_color=LISTENING_COLOR)
                ring_radius = BASE_RADIUS + 4 * math.sin(t * 3)
            elif state == "pensando":
                ring_radius = BASE_RADIUS
            else:
                ring_radius = BASE_RADIUS + 5 * math.sin(t * 1.2)

            pygame.draw.circle(screen, RING_COLOR, CENTER, int(ring_radius))
            pygame.draw.circle(screen, RING_BORDER_COLOR, CENTER, int(ring_radius), width=3)

            if state == "pensando":
                self._draw_loading(screen, t)

            agent_name = self._read_agent_name()
            if agent_name:
                text = agent_name_font.render(f"Agente: {agent_name}", True, (255, 255, 255))
                screen.blit(text, (WIDTH - text.get_width() - 10, 10))

            pygame.display.flip()
            dt = clock.tick(60) / 1000.0
            t += dt

        pygame.display.quit()

    def _draw_waves(self, screen, t, phases, intensity, single_color=None):
        surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for i, phase in enumerate(phases):
            variation = math.sin(t * 4 + phase * 3) * 0.5 + math.sin(t * 7 + phase) * 0.5
            length = (18 + 34 * abs(variation)) * intensity
            if length < 1:
                continue

            angle = phase + t * 0.15
            x1 = CENTER[0] + math.cos(angle) * BASE_RADIUS
            y1 = CENTER[1] + math.sin(angle) * BASE_RADIUS
            x2 = CENTER[0] + math.cos(angle) * (BASE_RADIUS + length)
            y2 = CENTER[1] + math.sin(angle) * (BASE_RADIUS + length)

            color = single_color or WAVE_COLORS[i % len(WAVE_COLORS)]
            alpha = int(90 + 120 * intensity)
            pygame.draw.line(surface, (*color, alpha), (x1, y1), (x2, y2), 3)

        screen.blit(surface, (0, 0))

    def _draw_loading(self, screen, t):
        orbit_radius = 22
        dot_radius = 6
        speed = 3.0
        for i in range(NUM_LOADING_DOTS):
            phase = i * (2 * math.pi / NUM_LOADING_DOTS)
            angle = t * speed + phase
            x = CENTER[0] + math.cos(angle) * orbit_radius
            y = CENTER[1] + math.sin(angle) * orbit_radius
            size = dot_radius * (0.4 + 0.6 * ((math.sin(angle - t * speed) + 1) / 2))
            brightness = 0.3 + 0.7 * (i / NUM_LOADING_DOTS)
            color = tuple(int(c * brightness) for c in LOADING_COLOR)
            pygame.draw.circle(screen, color, (int(x), int(y)), max(1, int(size)))


if __name__ == "__main__":
    # standalone test: cycles through states every 3 seconds
    panel = VisualPanel()
    panel.start()
    states = ["dormindo", "ouvindo", "pensando", "falando"]
    i = 0
    while panel._running.is_set():
        panel.set_state(states[i % len(states)])
        i += 1
        time.sleep(3)

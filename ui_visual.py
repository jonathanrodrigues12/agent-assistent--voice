"""
Painel visual do assistente: uma janela com um circulo animado.
Parado = respiracao suave. Ouvindo = pulso azul. Falando = ondas coloridas
irradiando, inspirado num waveform circular.
Roda numa thread separada, controlado por PainelVisual.definir_estado().
"""

import math
import threading
import time

import pygame

LARGURA, ALTURA = 420, 420
CENTRO = (LARGURA // 2, ALTURA // 2)
RAIO_BASE = 110

COR_FUNDO = (12, 12, 20)
COR_ANEL = (20, 20, 35)
COR_ANEL_BORDA = (235, 235, 245)

CORES_ONDA = [
    (255, 111, 97),   # coral
    (79, 168, 216),   # azul
    (124, 92, 191),   # roxo
]
COR_OUVINDO = (79, 168, 216)

NUM_RAIOS = 64


class PainelVisual:
    def __init__(self, titulo="Assistente"):
        self._estado = "dormindo"  # dormindo | ouvindo | falando
        self._lock = threading.Lock()
        self._rodando = threading.Event()
        self._rodando.set()
        self._titulo = titulo
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def iniciar(self):
        self._thread.start()

    def definir_estado(self, estado):
        with self._lock:
            self._estado = estado

    def _ler_estado(self):
        with self._lock:
            return self._estado

    def encerrar(self):
        self._rodando.clear()

    def _loop(self):
        pygame.init()
        tela = pygame.display.set_mode((LARGURA, ALTURA))
        pygame.display.set_caption(self._titulo)
        relogio = pygame.time.Clock()

        t = 0.0
        fases = [i * (2 * math.pi / NUM_RAIOS) for i in range(NUM_RAIOS)]

        while self._rodando.is_set():
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self._rodando.clear()

            estado = self._ler_estado()
            tela.fill(COR_FUNDO)

            if estado == "falando":
                self._desenhar_ondas(tela, t, fases, intensidade=1.0)
                raio_anel = RAIO_BASE + 6 * math.sin(t * 6)
            elif estado == "ouvindo":
                self._desenhar_ondas(tela, t, fases, intensidade=0.35, cor_unica=COR_OUVINDO)
                raio_anel = RAIO_BASE + 4 * math.sin(t * 3)
            else:
                raio_anel = RAIO_BASE + 5 * math.sin(t * 1.2)

            pygame.draw.circle(tela, COR_ANEL, CENTRO, int(raio_anel))
            pygame.draw.circle(tela, COR_ANEL_BORDA, CENTRO, int(raio_anel), width=3)

            pygame.display.flip()
            dt = relogio.tick(60) / 1000.0
            t += dt

        pygame.quit()

    def _desenhar_ondas(self, tela, t, fases, intensidade, cor_unica=None):
        superficie = pygame.Surface((LARGURA, ALTURA), pygame.SRCALPHA)
        for i, fase in enumerate(fases):
            variacao = math.sin(t * 4 + fase * 3) * 0.5 + math.sin(t * 7 + fase) * 0.5
            comprimento = (18 + 34 * abs(variacao)) * intensidade
            if comprimento < 1:
                continue

            angulo = fase + t * 0.15
            x1 = CENTRO[0] + math.cos(angulo) * RAIO_BASE
            y1 = CENTRO[1] + math.sin(angulo) * RAIO_BASE
            x2 = CENTRO[0] + math.cos(angulo) * (RAIO_BASE + comprimento)
            y2 = CENTRO[1] + math.sin(angulo) * (RAIO_BASE + comprimento)

            cor = cor_unica or CORES_ONDA[i % len(CORES_ONDA)]
            alpha = int(90 + 120 * intensidade)
            pygame.draw.line(superficie, (*cor, alpha), (x1, y1), (x2, y2), 3)

        tela.blit(superficie, (0, 0))


if __name__ == "__main__":
    # teste isolado: alterna estados a cada 3 segundos
    painel = PainelVisual()
    painel.iniciar()
    estados = ["dormindo", "ouvindo", "falando"]
    i = 0
    while painel._rodando.is_set():
        painel.definir_estado(estados[i % len(estados)])
        i += 1
        time.sleep(3)

"""
Assistente de voz - conversa continua com o Qwen3 8B via Ollama.
Fala -> Whisper local (transcricao) -> Ollama (qwen3:8b, em streaming) -> Edge TTS (fala a resposta)
A resposta comeca a ser falada assim que a primeira frase sai do modelo,
sem esperar o texto inteiro ficar pronto.
Transcricao e o modelo rodam offline; a sintese de voz (Edge TTS) usa internet.
Suporta interrupcao: falar enquanto a IA esta respondendo corta a fala dela na hora.
"""

import asyncio
import re
import tempfile
import threading
import os

import pyaudio
import torch
import speech_recognition as sr
import edge_tts
import pygame
import ollama
from silero_vad import load_silero_vad

MODEL = "qwen3:8b"
WHISPER_MODEL = "small"  # tiny, base, small, medium (maior = mais preciso, mais lento)
WHISPER_DEVICE = "cuda"  # "cuda" usa a GPU (rapido), "cpu" se nao tiver GPU disponivel
LANGUAGE = "portuguese"
VOICE = "pt-BR-FranciscaNeural"
NOME_MICROFONE = "fifine"  # trecho do nome do microfone a ser usado (ver lista abaixo)
VAD_LIMIAR = 0.6  # confianca minima (0-1) de que e fala humana, nao ruido
VAD_FRAMES_CONSECUTIVOS = 4  # frames seguidos de fala pra confirmar interrupcao (evita falso positivo)
VAD_TAMANHO_FRAME = 512  # amostras por frame, exigido pelo Silero VAD em 16kHz
NUM_PREDICT = 1024  # limite de tokens da resposta

SYSTEM_PROMPT = (
    "Voce e um assistente de voz. Converse normalmente sobre qualquer assunto "
    "que o usuario trouxer. Voce tambem tem bastante conhecimento de "
    "desenvolvimento Node.js e pode aprofundar tecnicamente quando o assunto "
    "for programacao, mas nao precisa puxar pra Node.js em perguntas que nao "
    "tem nada a ver com isso. "
    "Responda sempre em portugues do Brasil, de forma natural e completa, "
    "como numa conversa falada. Pode desenvolver o raciocinio e dar detalhes "
    "tecnicos quando fizer sentido, mas evite listas numeradas, formatacao "
    "markdown e blocos de codigo, ja que a resposta sera lida em voz alta - "
    "descreva o codigo em palavras em vez de escreve-lo literalmente. "
    "Nunca use emojis ou emoticons em nenhuma resposta."
)

TAMANHO_MINIMO_BLOCO = 80  # caracteres; agrupa frases curtas pra evitar pausas artificiais
PADRAO_FIM_DE_FRASE = re.compile(r"(?<=[.!?])\s+")
PADRAO_EMOJI = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
PADRAO_MARKDOWN = re.compile(r"[*_`#]")


def _achar_indice_microfone(trecho_nome):
    nomes = sr.Microphone.list_microphone_names()
    for indice, nome in enumerate(nomes):
        if trecho_nome.lower() in nome.lower():
            return indice
    return None


def _limpar_para_fala(texto):
    texto = PADRAO_EMOJI.sub("", texto)
    texto = PADRAO_MARKDOWN.sub("", texto)
    return texto.strip()


async def _sintetizar(texto, caminho):
    comunicador = edge_tts.Communicate(texto, VOICE)
    await comunicador.save(caminho)


class MonitorInterrupcao:
    """Escuta o microfone em uma thread separada enquanto a IA fala.
    Usa o Silero VAD (deteccao de voz humana) para so interromper quando
    alguem realmente esta falando, ignorando ruidos como latidos, batidas, etc."""

    _modelo_vad = None

    def __init__(self, indice_mic):
        self.indice_mic = indice_mic
        self.evento_interrupcao = threading.Event()
        self._evento_parar = threading.Event()
        self._thread = None
        if MonitorInterrupcao._modelo_vad is None:
            MonitorInterrupcao._modelo_vad = load_silero_vad()

    def _loop(self):
        modelo = MonitorInterrupcao._modelo_vad
        modelo.reset_states()

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=self.indice_mic,
            frames_per_buffer=VAD_TAMANHO_FRAME,
        )
        frames_seguidos = 0
        try:
            while not self._evento_parar.is_set():
                dados = stream.read(VAD_TAMANHO_FRAME, exception_on_overflow=False)
                amostras = torch.frombuffer(bytearray(dados), dtype=torch.int16).float() / 32768.0
                with torch.no_grad():
                    probabilidade = modelo(amostras, 16000).item()

                if probabilidade > VAD_LIMIAR:
                    frames_seguidos += 1
                    if frames_seguidos >= VAD_FRAMES_CONSECUTIVOS:
                        self.evento_interrupcao.set()
                        break
                else:
                    frames_seguidos = 0
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def iniciar(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def parar(self):
        self._evento_parar.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


async def _tocar(caminho, indice_mic):
    monitor = MonitorInterrupcao(indice_mic)
    monitor.iniciar()

    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play()
    try:
        while pygame.mixer.music.get_busy():
            if monitor.evento_interrupcao.is_set():
                pygame.mixer.music.stop()
                return True
            await asyncio.sleep(0.05)
    finally:
        pygame.mixer.music.unload()
        monitor.parar()

    return False


async def _tokens_ollama(historico):
    """Consome o stream do Ollama sem bloquear o event loop: cada chamada
    bloqueante ao gerador roda numa thread separada via executor."""
    loop = asyncio.get_event_loop()

    def _iniciar_stream():
        return ollama.chat(
            model=MODEL,
            messages=historico,
            think=False,
            stream=True,
            options={"num_predict": NUM_PREDICT},
        )

    iterador = await loop.run_in_executor(None, _iniciar_stream)

    def _proximo():
        try:
            return next(iterador)
        except StopIteration:
            return None

    while True:
        pedaco = await loop.run_in_executor(None, _proximo)
        if pedaco is None:
            break
        yield pedaco["message"]["content"]


async def _responder_e_falar(historico, indice_mic):
    """Consome a resposta do Ollama em streaming, sintetizando e falando
    cada frase assim que ela fica pronta, em paralelo com a geracao do resto."""
    fila = asyncio.Queue(maxsize=2)
    texto_completo = []

    async def produtor():
        buffer = ""
        print("Assistente: ", end="", flush=True)
        async for pedaco in _tokens_ollama(historico):
            print(pedaco, end="", flush=True)
            texto_completo.append(pedaco)
            buffer += pedaco
            while True:
                m = PADRAO_FIM_DE_FRASE.search(buffer)
                if not m or len(buffer[: m.start() + 1]) < TAMANHO_MINIMO_BLOCO:
                    break
                frase, buffer = buffer[: m.end()], buffer[m.end():]
                frase_limpa = _limpar_para_fala(frase)
                if frase_limpa:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        caminho = f.name
                    await _sintetizar(frase_limpa, caminho)
                    await fila.put(caminho)
        print()
        frase_limpa = _limpar_para_fala(buffer)
        if frase_limpa:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                caminho = f.name
            await _sintetizar(frase_limpa, caminho)
            await fila.put(caminho)
        await fila.put(None)

    tarefa_produtor = asyncio.create_task(produtor())
    interrompido = False
    try:
        while True:
            caminho = await fila.get()
            if caminho is None:
                break
            interrompido = await _tocar(caminho, indice_mic)
            os.remove(caminho)
            if interrompido:
                break
    finally:
        tarefa_produtor.cancel()
        try:
            await tarefa_produtor
        except (asyncio.CancelledError, Exception):
            pass

    return "".join(texto_completo).strip(), interrompido


def responder_e_falar(historico, indice_mic):
    return asyncio.run(_responder_e_falar(historico, indice_mic))


def main():
    pygame.mixer.init()
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0  # segundos de silencio para considerar que a frase acabou
    recognizer.non_speaking_duration = 0.5

    indice_mic = _achar_indice_microfone(NOME_MICROFONE)
    if indice_mic is None:
        print(f'Microfone "{NOME_MICROFONE}" nao encontrado, usando o padrao do sistema.')
        microfone = sr.Microphone()
    else:
        nome = sr.Microphone.list_microphone_names()[indice_mic]
        print(f"Usando microfone: [{indice_mic}] {nome}")
        microfone = sr.Microphone(device_index=indice_mic)

    historico = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Calibrando ruido ambiente... fique em silencio por um instante.")
    with microfone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
    print(f"Limiar de energia calibrado: {recognizer.energy_threshold:.0f}")

    print("Pronto! Pode falar quando quiser (Ctrl+C para sair). Pode interromper a fala dela a qualquer momento.\n")

    while True:
        try:
            with microfone as source:
                print("Ouvindo...")
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=20)

            print("Transcrevendo...")
            texto_usuario = recognizer.recognize_whisper(
                audio,
                model=WHISPER_MODEL,
                language=LANGUAGE,
                load_options={"device": WHISPER_DEVICE},
            ).strip()

            if not texto_usuario:
                continue

            print(f"Voce: {texto_usuario}")
            historico.append({"role": "user", "content": texto_usuario})

            texto_resposta, interrompido = responder_e_falar(historico, indice_mic)
            historico.append({"role": "assistant", "content": texto_resposta})
            if interrompido:
                print("(interrompido)")

        except sr.UnknownValueError:
            continue
        except sr.WaitTimeoutError:
            continue
        except KeyboardInterrupt:
            print("\nEncerrando.")
            break


if __name__ == "__main__":
    main()

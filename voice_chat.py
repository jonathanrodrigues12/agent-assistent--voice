"""
Assistente de voz - conversa continua com o Qwen3 8B via Ollama.
Fala -> Whisper local (transcricao) -> Ollama (qwen3:8b, em streaming) -> Edge TTS (fala a resposta)
A resposta comeca a ser falada assim que a primeira frase sai do modelo,
sem esperar o texto inteiro ficar pronto.
Transcricao e o modelo rodam offline; a sintese de voz (Edge TTS) usa internet.
Suporta interrupcao: falar enquanto a IA esta respondendo corta a fala dela na hora.
"""

import asyncio
import audioop
import json
import re
import tempfile
import threading
import os
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pyaudio
import torch
import whisper
import speech_recognition as sr
import edge_tts
import pygame
import pygame.sndarray
import ollama
from silero_vad import load_silero_vad

from ui_visual import PainelVisual

_painel = None
_som_carregamento = None
_canal_carregamento = None


def _definir_estado_visual(estado):
    if _painel is not None:
        _painel.definir_estado(estado)


def _gerar_som_carregamento():
    taxa = 44100
    duracao_som = 0.12
    duracao_silencio = 0.35
    freq = 523.25  # C5, som suave e curto

    t_som = np.linspace(0, duracao_som, int(taxa * duracao_som), False)
    envelope = np.sin(np.pi * t_som / duracao_som)  # fade in/out, evita clique
    onda = np.sin(2 * np.pi * freq * t_som) * envelope * 0.15  # volume baixo
    silencio = np.zeros(int(taxa * duracao_silencio))

    amostra = np.concatenate([onda, silencio])
    amostra_int16 = (amostra * 32767).astype(np.int16)
    estereo = np.column_stack([amostra_int16, amostra_int16])
    return pygame.sndarray.make_sound(estereo)


def _iniciar_som_carregamento():
    global _som_carregamento, _canal_carregamento
    if _som_carregamento is None:
        _som_carregamento = _gerar_som_carregamento()
    _canal_carregamento = _som_carregamento.play(loops=-1)


def _parar_som_carregamento():
    global _canal_carregamento
    if _canal_carregamento is not None:
        _canal_carregamento.stop()
        _canal_carregamento = None


MODEL = "qwen3:8b"
WHISPER_MODEL = "small"  # tiny, base, small, medium (maior = mais preciso, mais lento)
WHISPER_DEVICE = "cuda"  # "cuda" usa a GPU (rapido), "cpu" se nao tiver GPU disponivel
LANGUAGE = "portuguese"
VOICE = "pt-BR-FranciscaNeural"
NOME_MICROFONE = "fifine"  # trecho do nome do microfone a ser usado (ver lista abaixo)
VAD_LIMIAR = 0.6  # confianca minima (0-1) de que e fala humana, nao ruido
VAD_FRAMES_CONSECUTIVOS = 4  # frames seguidos de fala pra confirmar interrupcao (evita falso positivo)
VAD_TAMANHO_FRAME = 512  # amostras por frame, exigido pelo Silero VAD em 16kHz
MULTIPLICADOR_VOLUME_INTERRUPCAO = 3.0  # volume minimo (x limiar de ruido ambiente) pra contar como voce falando de perto
NUM_PREDICT = 1024  # limite de tokens da resposta

CAMINHO_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
FRASES_PARAR = [
    "pare de ouvir",
    "pode dormir",
    "para de escutar",
    "para de ouvir",
    "obrigado",
    "obrigada",
    "pare",
    "parar",
    "pode parar",
    "chega",
    "silencio",
]
LIMIAR_SIMILARIDADE_NOME = 0.55  # 0-1; quanto menor, mais tolerante a erros de transcricao do nome

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
TAMANHO_MAXIMO_BLOCO = 220  # forca quebra mesmo sem pontuacao, pra nao acumular o texto todo
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


_modelo_whisper = None


def _carregar_modelo_whisper():
    global _modelo_whisper
    if _modelo_whisper is None:
        print("Carregando modelo Whisper...")
        _modelo_whisper = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)
    return _modelo_whisper


def transcrever(audio_data):
    """Transcreve um sr.AudioData chamando o whisper direto, com o modelo
    ja carregado em memoria (a wrapper do speech_recognition recarrega o
    modelo do zero a cada chamada, o que deixava tudo muito lento)."""
    modelo = _carregar_modelo_whisper()
    dados_brutos = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
    amostras = np.frombuffer(dados_brutos, dtype=np.int16).astype(np.float32) / 32768.0
    resultado = modelo.transcribe(
        amostras, language=LANGUAGE, fp16=(WHISPER_DEVICE == "cuda")
    )
    return resultado["text"].strip()


def _normalizar(texto):
    """Minusculas e sem acento, pra comparar frases faladas com mais tolerancia."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sem_acento.lower().strip()


def _carregar_ou_criar_config(recognizer, microfone):
    if os.path.exists(CAMINHO_CONFIG):
        with open(CAMINHO_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("nome_assistente"):
            return config

    print("Primeira vez rodando o assistente. Vamos dar um nome pra ela por voz.")
    falar_curto("Ola! Ainda nao tenho um nome. Diga, em uma palavra so, o nome que voce quer me dar.")

    nome = ""
    while not nome:
        _definir_estado_visual("ouvindo")
        with microfone as source:
            print("Ouvindo o nome...")
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
        texto = transcrever(audio)
        nome = texto.strip(" .,!?").split()[0].capitalize() if texto.strip() else ""
        if not nome:
            falar_curto("Nao entendi, pode repetir o nome?")

    config = {"nome_assistente": nome}
    with open(CAMINHO_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    falar_curto(f"Combinado! Pode me chamar de {nome} a partir de agora.")
    return config


def _similaridade(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _extrair_comando_apos_palavra_chave(texto, palavra_chave):
    """Procura a palavra-chave no texto por aproximacao (o Whisper erra
    nomes curtos com frequencia, ex: 'Zoe' pode sair como 'Zoi' ou 'Zule').
    Se achar uma palavra parecida o suficiente, devolve o que vier depois
    dela (o comando). Se nao achar nada parecido, devolve None."""
    chave_normalizada = _normalizar(palavra_chave)
    palavras = texto.split()

    for i, palavra in enumerate(palavras):
        palavra_limpa = re.sub(r"[^\w]", "", _normalizar(palavra))
        if not palavra_limpa:
            continue
        if _similaridade(palavra_limpa, chave_normalizada) >= LIMIAR_SIMILARIDADE_NOME:
            resto = " ".join(palavras[i + 1:]).strip(" ,.:;!?-")
            return resto

    return None


PADROES_PARAR = [
    re.compile(r"\b" + re.escape(_normalizar(frase)) + r"\b") for frase in FRASES_PARAR
]


def _contem_frase_de_parar(texto):
    texto_normalizado = _normalizar(texto)
    return any(padrao.search(texto_normalizado) for padrao in PADROES_PARAR)


def _limpar_para_fala(texto):
    texto = PADRAO_EMOJI.sub("", texto)
    texto = PADRAO_MARKDOWN.sub("", texto)
    return texto.strip()


def _extrair_blocos_prontos(buffer):
    """Retira do buffer os trechos ja prontos pra falar: por pontuacao de fim
    de frase, ou por tamanho maximo (forca quebra mesmo sem pontuacao, pra
    nao acumular o texto inteiro esperando um ponto final)."""
    blocos = []
    while True:
        m = PADRAO_FIM_DE_FRASE.search(buffer)
        if m and len(buffer[: m.start() + 1]) >= TAMANHO_MINIMO_BLOCO:
            blocos.append(buffer[: m.end()])
            buffer = buffer[m.end():]
            continue

        if len(buffer) >= TAMANHO_MAXIMO_BLOCO:
            pos_corte = buffer.rfind(" ", 0, TAMANHO_MAXIMO_BLOCO)
            if pos_corte <= 0:
                pos_corte = TAMANHO_MAXIMO_BLOCO
            blocos.append(buffer[:pos_corte])
            buffer = buffer[pos_corte:].lstrip()
            continue

        break

    return blocos, buffer


async def _sintetizar(texto, caminho):
    comunicador = edge_tts.Communicate(texto, VOICE)
    await comunicador.save(caminho)


async def _falar_curto_async(texto):
    """Fala uma frase curta de confirmacao, sem streaming nem interrupcao
    (usado para 'oi, pode falar' e 'ok, vou dormir')."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        caminho = f.name
    await _sintetizar(texto, caminho)
    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.05)
    pygame.mixer.music.unload()
    os.remove(caminho)


def falar_curto(texto):
    print(f"Assistente: {texto}")
    _definir_estado_visual("falando")
    asyncio.run(_falar_curto_async(texto))


class MonitorInterrupcao:
    """Escuta o microfone em uma thread separada enquanto a IA fala.
    So interrompe quando dois sinais batem ao mesmo tempo: o Silero VAD
    confirma que e voz humana (nao ruido tipo latido, batida) E o volume
    esta alto o suficiente pra ser alguem falando perto do microfone
    (filtra conversas ao fundo, que chegam mais baixas)."""

    _modelo_vad = None

    def __init__(self, indice_mic, limiar_volume):
        self.indice_mic = indice_mic
        self.limiar_volume = limiar_volume
        self.evento_interrupcao = threading.Event()
        self._evento_parar = threading.Event()
        self._thread = None
        self.audio_capturado = None
        if MonitorInterrupcao._modelo_vad is None:
            MonitorInterrupcao._modelo_vad = load_silero_vad()

    def _probabilidade_fala(self, modelo, dados):
        rms = audioop.rms(dados, 2)
        if rms < self.limiar_volume:
            return 0.0
        amostras = torch.frombuffer(bytearray(dados), dtype=torch.int16).float() / 32768.0
        with torch.no_grad():
            return modelo(amostras, 16000).item()

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
        try:
            # fase 1: espera confirmar que e voce falando (nao ruido/fundo)
            frames_seguidos = 0
            buffer_captura = bytearray()
            while not self._evento_parar.is_set():
                dados = stream.read(VAD_TAMANHO_FRAME, exception_on_overflow=False)
                probabilidade = self._probabilidade_fala(modelo, dados)

                if probabilidade > VAD_LIMIAR:
                    buffer_captura.extend(dados)
                    frames_seguidos += 1
                    if frames_seguidos >= VAD_FRAMES_CONSECUTIVOS:
                        self.evento_interrupcao.set()
                        break
                else:
                    frames_seguidos = 0
                    buffer_captura.clear()

            if not self.evento_interrupcao.is_set():
                return

            # fase 2: continua gravando ate voce parar de falar (ou estourar o tempo maximo)
            frames_silencio = 0
            frames_silencio_necessarios = int(1.2 * 16000 / VAD_TAMANHO_FRAME)
            max_frames = int(20 * 16000 / VAD_TAMANHO_FRAME)  # limite de seguranca: 20s
            for _ in range(max_frames):
                dados = stream.read(VAD_TAMANHO_FRAME, exception_on_overflow=False)
                buffer_captura.extend(dados)
                probabilidade = self._probabilidade_fala(modelo, dados)
                if probabilidade > VAD_LIMIAR:
                    frames_silencio = 0
                else:
                    frames_silencio += 1
                    if frames_silencio >= frames_silencio_necessarios:
                        break

            self.audio_capturado = bytes(buffer_captura)
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


async def _tocar(caminho, indice_mic, limiar_volume):
    """Toca um trecho de audio. Se for interrompido, espera o monitor
    terminar de capturar o que a pessoa esta falando e devolve esse audio
    junto, pra nao precisar pedir pra repetir."""
    monitor = MonitorInterrupcao(indice_mic, limiar_volume)
    monitor.iniciar()

    interrompido = False
    pygame.mixer.music.load(caminho)
    pygame.mixer.music.play()
    try:
        while pygame.mixer.music.get_busy():
            if monitor.evento_interrupcao.is_set():
                pygame.mixer.music.stop()
                interrompido = True
                break
            await asyncio.sleep(0.05)
    finally:
        pygame.mixer.music.unload()

    if interrompido:
        while monitor._thread.is_alive():
            await asyncio.sleep(0.05)
        return True, monitor.audio_capturado

    monitor.parar()
    return False, None


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


async def _responder_e_falar(historico, indice_mic, limiar_volume):
    """Consome a resposta do Ollama em streaming, sintetizando e falando
    cada frase assim que ela fica pronta, em paralelo com a geracao do resto."""
    _definir_estado_visual("pensando")
    _iniciar_som_carregamento()
    fila = asyncio.Queue(maxsize=2)

    async def produtor():
        buffer = ""
        print("Assistente: ", end="", flush=True)
        async for pedaco in _tokens_ollama(historico):
            print(pedaco, end="", flush=True)
            buffer += pedaco
            blocos, buffer = _extrair_blocos_prontos(buffer)
            for bloco in blocos:
                frase_limpa = _limpar_para_fala(bloco)
                if frase_limpa:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        caminho = f.name
                    await _sintetizar(frase_limpa, caminho)
                    await fila.put((caminho, frase_limpa))
        print()
        frase_limpa = _limpar_para_fala(buffer)
        if frase_limpa:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                caminho = f.name
            await _sintetizar(frase_limpa, caminho)
            await fila.put((caminho, frase_limpa))
        await fila.put(None)

    tarefa_produtor = asyncio.create_task(produtor())
    interrompido = False
    audio_capturado = None
    primeiro_audio = True
    texto_falado = []
    try:
        while True:
            item = await fila.get()
            if item is None:
                break
            caminho, frase_limpa = item
            if primeiro_audio:
                _parar_som_carregamento()
                _definir_estado_visual("falando")
                primeiro_audio = False
            interrompido, audio_capturado = await _tocar(caminho, indice_mic, limiar_volume)
            os.remove(caminho)
            if not interrompido:
                texto_falado.append(frase_limpa)
            if interrompido:
                break
    finally:
        _parar_som_carregamento()
        tarefa_produtor.cancel()
        try:
            await tarefa_produtor
        except (asyncio.CancelledError, Exception):
            pass

    return " ".join(texto_falado).strip(), interrompido, audio_capturado


def responder_e_falar(historico, indice_mic, limiar_volume):
    return asyncio.run(_responder_e_falar(historico, indice_mic, limiar_volume))


def main():
    global _painel
    pygame.mixer.init()
    _painel = PainelVisual("Assistente de Voz")
    _painel.iniciar()
    _carregar_modelo_whisper()

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
    limiar_volume_interrupcao = recognizer.energy_threshold * MULTIPLICADOR_VOLUME_INTERRUPCAO

    config = _carregar_ou_criar_config(recognizer, microfone)
    nome_assistente = config["nome_assistente"]

    modo = "dormindo"
    audio_pendente = None
    print(f'\nPronto! Diga "{nome_assistente}" pra ativar. (Ctrl+C para sair)\n')

    while True:
        try:
            if audio_pendente is not None:
                audio = audio_pendente
                audio_pendente = None
            else:
                _definir_estado_visual("ouvindo" if modo == "ativo" else "dormindo")
                with microfone as source:
                    print("Ouvindo..." if modo == "ativo" else f'Esperando "{nome_assistente}"...')
                    audio = recognizer.listen(source, timeout=15, phrase_time_limit=20)

            print("Transcrevendo...")
            texto_usuario = transcrever(audio)

            if not texto_usuario:
                continue

            if modo == "dormindo":
                comando = _extrair_comando_apos_palavra_chave(texto_usuario, nome_assistente)
                if comando is None:
                    print(f'(ouvido: "{texto_usuario}" - nao reconheci o nome "{nome_assistente}")')
                    continue  # nao disse o nome, ignora e continua dormindo

                modo = "ativo"
                if not comando:
                    falar_curto("Oi, pode falar.")
                    continue
                texto_usuario = comando

            print(f"Voce: {texto_usuario}")

            if _contem_frase_de_parar(texto_usuario):
                falar_curto("Ok, pode me chamar quando precisar.")
                modo = "dormindo"
                continue

            historico.append({"role": "user", "content": texto_usuario})

            texto_resposta, interrompido, audio_capturado = responder_e_falar(
                historico, indice_mic, limiar_volume_interrupcao
            )
            historico.append({"role": "assistant", "content": texto_resposta})
            if interrompido:
                print("(interrompido)")
                if audio_capturado:
                    audio_pendente = sr.AudioData(audio_capturado, 16000, 2)

        except sr.UnknownValueError:
            continue
        except sr.WaitTimeoutError:
            continue
        except KeyboardInterrupt:
            print("\nEncerrando.")
            break


if __name__ == "__main__":
    main()

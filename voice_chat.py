"""
Voice assistant - continuous conversation with Qwen3 8B via Ollama.
Speech -> local Whisper (transcription) -> Ollama (qwen3:8b, streaming) -> Edge TTS (speaks the response)
The response starts being spoken as soon as the first sentence comes out of the model,
without waiting for the whole text to be ready.
Transcription and the model run offline; speech synthesis (Edge TTS) needs internet.
Supports interruption: speaking while the AI is responding cuts off its speech immediately.
"""

import asyncio
import audioop
import json
import re
import tempfile
import threading
import time
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

from ui_visual import VisualPanel

_panel = None
_loading_sound = None
_loading_channel = None


def _set_visual_state(state):
    if _panel is not None:
        _panel.set_state(state)


def _generate_loading_sound():
    sample_rate = 44100
    sound_duration = 0.12
    silence_duration = 0.35
    freq = 523.25  # C5, soft and short sound

    t_sound = np.linspace(0, sound_duration, int(sample_rate * sound_duration), False)
    envelope = np.sin(np.pi * t_sound / sound_duration)  # fade in/out, avoids click
    wave = np.sin(2 * np.pi * freq * t_sound) * envelope * 0.15  # low volume
    silence = np.zeros(int(sample_rate * silence_duration))

    sample = np.concatenate([wave, silence])
    sample_int16 = (sample * 32767).astype(np.int16)
    stereo = np.column_stack([sample_int16, sample_int16])
    return pygame.sndarray.make_sound(stereo)


def _start_loading_sound():
    global _loading_sound, _loading_channel
    if _loading_sound is None:
        _loading_sound = _generate_loading_sound()
    _loading_channel = _loading_sound.play(loops=-1)


def _stop_loading_sound():
    global _loading_channel
    if _loading_channel is not None:
        _loading_channel.stop()
        _loading_channel = None


MODEL = "qwen3:8b"
WHISPER_MODEL = "small"  # tiny, base, small, medium (bigger = more accurate, slower)
WHISPER_DEVICE = "cuda"  # "cuda" uses the GPU (fast), "cpu" if no GPU is available
BEAM_SIZE = 1  # 1 = greedy, much faster; higher values (e.g. 5) are more accurate but slower
LANGUAGE = "portuguese"
FEMALE_VOICE = "pt-BR-FranciscaNeural"
MALE_VOICE = "pt-BR-AntonioNeural"
VOICE = FEMALE_VOICE  # default until config is loaded
MIC_NAME = "fifine"  # substring of the microphone name to use (see list below)
VAD_THRESHOLD = 0.6  # minimum confidence (0-1) that it's human speech, not noise
VAD_CONSECUTIVE_FRAMES = 4  # consecutive speech frames to confirm interruption (avoids false positives)
VAD_FRAME_SIZE = 512  # samples per frame, required by Silero VAD at 16kHz
INTERRUPTION_VOLUME_MULTIPLIER = 3.0  # minimum volume (x ambient noise threshold) to count as speaking up close
NUM_PREDICT = 1024  # response token limit

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
STOP_PHRASES = [
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
NAME_SIMILARITY_THRESHOLD = 0.55  # 0-1; the lower, the more tolerant to transcription errors in the name

CHANGE_VOICE_PHRASES = [
    "mudar a voz",
    "mudar voz",
    "trocar a voz",
    "trocar voz",
    "muda a voz",
    "muda voz",
]
CHANGE_NAME_PHRASES = [
    "mudar o nome",
    "mudar nome",
    "trocar o nome",
    "trocar nome",
    "muda o nome",
    "muda nome",
]

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
    "Nunca use emojis ou emoticons em nenhuma resposta. "
    "Quando o usuario pedir pra voce escrever, redigir ou criar algo (uma "
    "mensagem, um texto, um resumo, etc.), NAO fique so perguntando detalhes "
    "antes de comecar. Em vez disso, ja produza uma primeira versao completa "
    "usando o contexto que voce tem, mesmo que precise assumir alguma coisa "
    "razoavel. Depois de entregar essa primeira versao, pergunte se quer "
    "ajustar algo. So peca esclarecimento antes de comecar se for realmente "
    "impossivel prosseguir sem essa informacao."
)

MIN_CHUNK_SIZE = 80  # characters; groups short sentences to avoid artificial pauses
MIN_FIRST_CHUNK_SIZE = 10  # only for the first sentence, to start speaking faster
MAX_CHUNK_SIZE = 220  # forces a break even without punctuation, to avoid piling up the whole text
MAX_FIRST_CHUNK_SIZE = 60  # cuts the first chunk mid-sentence if it takes too long to finish
SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])\s+")
EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
MARKDOWN_PATTERN = re.compile(r"[*_`#]")


def _find_microphone_index(name_snippet):
    names = sr.Microphone.list_microphone_names()
    for index, name in enumerate(names):
        if name_snippet.lower() in name.lower():
            return index
    return None


_whisper_model = None


def _load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        _whisper_model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)
    return _whisper_model


def transcribe(audio_data):
    """Transcribes an sr.AudioData by calling whisper directly, with the model
    already loaded in memory (the speech_recognition wrapper reloads the
    model from scratch on every call, which made everything very slow)."""
    model = _load_whisper_model()
    raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
    samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    # silence padding at the start/end: significantly improves Whisper's
    # accuracy on short recordings (single-word phrases/commands)
    padding = np.zeros(int(16000 * 0.3), dtype=np.float32)
    samples = np.concatenate([padding, samples, padding])

    start = time.perf_counter()
    result = model.transcribe(
        samples,
        language=LANGUAGE,
        fp16=(WHISPER_DEVICE == "cuda"),
        beam_size=BEAM_SIZE,
    )
    print(f"[time] transcription: {time.perf_counter() - start:.2f}s")
    return result["text"].strip()


def _normalize(text):
    """Lowercase and accent-free, to compare spoken phrases with more tolerance."""
    without_accents = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return without_accents.lower().strip()


def _load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _update_config(key, value):
    config = _load_config()
    config[key] = value
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config


def _choose_voice(recognizer, microphone):
    """Plays a sample in each voice, asks which one the person prefers and
    persists the choice in config.json. Returns 'masculina' or 'feminina'."""
    global VOICE

    speak_short("Voce prefere a voz masculina", voice=MALE_VOICE)
    speak_short("ou voce prefere a voz feminina?", voice=FEMALE_VOICE)

    choice = None
    while choice is None:
        _set_visual_state("ouvindo")
        with microphone as source:
            print("Listening for the voice choice...")
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
        text = _normalize(transcribe(audio))
        if "masculin" in text:
            choice = "masculina"
        elif "feminin" in text:
            choice = "feminina"
        else:
            speak_short("Nao entendi, diga masculina ou feminina.", voice=FEMALE_VOICE)

    VOICE = MALE_VOICE if choice == "masculina" else FEMALE_VOICE
    _update_config("voz", choice)
    speak_short(f"Combinado, vou falar com a voz {choice}.")
    return choice


def _choose_name(recognizer, microphone):
    """Asks for the name by voice and persists the choice in config.json."""
    speak_short("Diga, em uma palavra so, o nome que voce quer me dar.")

    name = ""
    while not name:
        _set_visual_state("ouvindo")
        with microphone as source:
            print("Listening for the name...")
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
        text = transcribe(audio)
        name = text.strip(" .,!?").split()[0].capitalize() if text.strip() else ""
        if not name:
            speak_short("Nao entendi, pode repetir o nome?")

    _update_config("nome_assistente", name)
    speak_short(f"Combinado! Pode me chamar de {name} a partir de agora.")
    return name


def _load_or_create_config(recognizer, microphone):
    global VOICE

    config = _load_config()

    if config.get("voz"):
        VOICE = MALE_VOICE if config["voz"] == "masculina" else FEMALE_VOICE

    if config.get("nome_assistente") and config.get("voz"):
        return config

    print("First time running the assistant. Let's configure it by voice.")

    if not config.get("voz"):
        _choose_voice(recognizer, microphone)

    if not config.get("nome_assistente"):
        _choose_name(recognizer, microphone)

    return _load_config()


def _similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _extract_command_after_keyword(text, keyword):
    """Looks for the keyword in the text by approximate match (Whisper often
    misrecognizes short names, e.g. 'Zoe' may come out as 'Zoi' or 'Zule').
    If it finds a close enough match, returns whatever comes after it (the
    command). If nothing close enough is found, returns None."""
    normalized_keyword = _normalize(keyword)
    words = text.split()

    for i, word in enumerate(words):
        clean_word = re.sub(r"[^\w]", "", _normalize(word))
        if not clean_word:
            continue
        if _similarity(clean_word, normalized_keyword) >= NAME_SIMILARITY_THRESHOLD:
            rest = " ".join(words[i + 1:]).strip(" ,.:;!?-")
            return rest

    return None


def _is_just_the_name(text, name):
    """True if the user only called the assistant's name and said nothing
    else (e.g. 'Marcia?', just to get attention), so we can reply quickly
    with 'go ahead' instead of sending it to the LLM as if it were a request."""
    clean = text.strip(" .,!?-")
    if not clean:
        return False
    return _similarity(_normalize(clean), _normalize(name)) >= NAME_SIMILARITY_THRESHOLD


STOP_PATTERNS = [
    re.compile(r"\b" + re.escape(_normalize(phrase)) + r"\b") for phrase in STOP_PHRASES
]


def _contains_stop_phrase(text):
    normalized_text = _normalize(text)
    return any(pattern.search(normalized_text) for pattern in STOP_PATTERNS)


CHANGE_VOICE_PATTERNS = [
    re.compile(r"\b" + re.escape(_normalize(phrase)) + r"\b") for phrase in CHANGE_VOICE_PHRASES
]
CHANGE_NAME_PATTERNS = [
    re.compile(r"\b" + re.escape(_normalize(phrase)) + r"\b") for phrase in CHANGE_NAME_PHRASES
]


def _contains_change_voice_phrase(text):
    normalized_text = _normalize(text)
    return any(pattern.search(normalized_text) for pattern in CHANGE_VOICE_PATTERNS)


def _contains_change_name_phrase(text):
    normalized_text = _normalize(text)
    return any(pattern.search(normalized_text) for pattern in CHANGE_NAME_PATTERNS)


def _clean_for_speech(text):
    text = EMOJI_PATTERN.sub("", text)
    text = MARKDOWN_PATTERN.sub("", text)
    return text.strip()


def _extract_ready_chunks(buffer, min_size=MIN_CHUNK_SIZE, max_size=MAX_CHUNK_SIZE):
    """Pulls out of the buffer the chunks that are ready to speak: either by
    sentence-ending punctuation, or by max size (forces a break even without
    punctuation, to avoid piling up the whole text while waiting for a period).
    min_size/max_size can be reduced for the first sentence, to start speaking
    faster instead of waiting to accumulate a large chunk."""
    chunks = []
    while True:
        m = SENTENCE_END_PATTERN.search(buffer)
        if m and len(buffer[: m.start() + 1]) >= min_size:
            chunks.append(buffer[: m.end()])
            buffer = buffer[m.end():]
            min_size = MIN_CHUNK_SIZE
            max_size = MAX_CHUNK_SIZE
            continue

        if len(buffer) >= max_size:
            cut_pos = buffer.rfind(" ", 0, max_size)
            if cut_pos <= 0:
                cut_pos = max_size
            chunks.append(buffer[:cut_pos])
            buffer = buffer[cut_pos:].lstrip()
            min_size = MIN_CHUNK_SIZE
            max_size = MAX_CHUNK_SIZE
            continue

        break

    return chunks, buffer


async def _synthesize(text, path, voice=None):
    communicator = edge_tts.Communicate(text, voice or VOICE)
    await communicator.save(path)


async def _speak_short_async(text, voice=None):
    """Speaks a short confirmation sentence, without streaming or interruption
    (used for 'hi, go ahead' and 'ok, going to sleep')."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    await _synthesize(text, path, voice)
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.05)
    pygame.mixer.music.unload()
    os.remove(path)


def speak_short(text, voice=None):
    print(f"Assistant: {text}")
    _set_visual_state("falando")
    asyncio.run(_speak_short_async(text, voice))


class InterruptionMonitor:
    """Listens to the microphone in a separate thread while the AI is speaking.
    Only interrupts when two signals match at the same time: the Silero VAD
    confirms it's human speech (not noise like a bark or a bump) AND the
    volume is loud enough to be someone speaking close to the microphone
    (filters out background conversations, which arrive quieter)."""

    _vad_model = None

    def __init__(self, mic_index, volume_threshold):
        self.mic_index = mic_index
        self.volume_threshold = volume_threshold
        self.interruption_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None
        self.captured_audio = None
        if InterruptionMonitor._vad_model is None:
            InterruptionMonitor._vad_model = load_silero_vad()

    def _speech_probability(self, model, data):
        rms = audioop.rms(data, 2)
        if rms < self.volume_threshold:
            return 0.0
        samples = torch.frombuffer(bytearray(data), dtype=torch.int16).float() / 32768.0
        with torch.no_grad():
            return model(samples, 16000).item()

    def _loop(self):
        model = InterruptionMonitor._vad_model
        model.reset_states()

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=VAD_FRAME_SIZE,
        )
        try:
            # phase 1: wait to confirm it's you speaking (not noise/background)
            consecutive_frames = 0
            capture_buffer = bytearray()
            while not self._stop_event.is_set():
                data = stream.read(VAD_FRAME_SIZE, exception_on_overflow=False)
                probability = self._speech_probability(model, data)

                if probability > VAD_THRESHOLD:
                    capture_buffer.extend(data)
                    consecutive_frames += 1
                    if consecutive_frames >= VAD_CONSECUTIVE_FRAMES:
                        self.interruption_event.set()
                        break
                else:
                    consecutive_frames = 0
                    capture_buffer.clear()

            if not self.interruption_event.is_set():
                return

            # phase 2: keep recording until you stop talking (or the max time is hit)
            silence_frames = 0
            silence_frames_needed = int(1.2 * 16000 / VAD_FRAME_SIZE)
            max_frames = int(20 * 16000 / VAD_FRAME_SIZE)  # safety limit: 20s
            for _ in range(max_frames):
                data = stream.read(VAD_FRAME_SIZE, exception_on_overflow=False)
                capture_buffer.extend(data)
                probability = self._speech_probability(model, data)
                if probability > VAD_THRESHOLD:
                    silence_frames = 0
                else:
                    silence_frames += 1
                    if silence_frames >= silence_frames_needed:
                        break

            self.captured_audio = bytes(capture_buffer)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


async def _play(path, mic_index, volume_threshold):
    """Plays an audio chunk. If interrupted, waits for the monitor to finish
    capturing what the person is saying and returns that audio along with it,
    so there's no need to ask them to repeat."""
    monitor = InterruptionMonitor(mic_index, volume_threshold)
    monitor.start()

    interrupted = False
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    try:
        while pygame.mixer.music.get_busy():
            if monitor.interruption_event.is_set():
                pygame.mixer.music.stop()
                interrupted = True
                break
            await asyncio.sleep(0.05)
    finally:
        pygame.mixer.music.unload()

    if interrupted:
        while monitor._thread.is_alive():
            await asyncio.sleep(0.05)
        return True, monitor.captured_audio

    monitor.stop()
    return False, None


async def _ollama_tokens(history):
    """Consumes the Ollama stream without blocking the event loop: each
    blocking call to the generator runs in a separate thread via the executor."""
    loop = asyncio.get_event_loop()

    def _start_stream():
        return ollama.chat(
            model=MODEL,
            messages=history,
            think=False,
            stream=True,
            options={"num_predict": NUM_PREDICT},
        )

    iterator = await loop.run_in_executor(None, _start_stream)

    def _next():
        try:
            return next(iterator)
        except StopIteration:
            return None

    while True:
        chunk = await loop.run_in_executor(None, _next)
        if chunk is None:
            break
        yield chunk["message"]["content"]


async def _respond_and_speak(history, mic_index, volume_threshold):
    """Consumes the Ollama response in streaming, synthesizing and speaking
    each sentence as soon as it's ready, in parallel with generating the rest."""
    _set_visual_state("pensando")
    _start_loading_sound()
    queue = asyncio.Queue(maxsize=2)
    start_total = time.perf_counter()

    async def producer():
        buffer = ""
        first_chunk = True
        first_token = True
        print("Assistant: ", end="", flush=True)
        async for chunk in _ollama_tokens(history):
            if first_token:
                print(f"\n[time] first token from ollama: {time.perf_counter() - start_total:.2f}s")
                first_token = False
            print(chunk, end="", flush=True)
            buffer += chunk
            min_size = MIN_FIRST_CHUNK_SIZE if first_chunk else MIN_CHUNK_SIZE
            max_size = MAX_FIRST_CHUNK_SIZE if first_chunk else MAX_CHUNK_SIZE
            chunks, buffer = _extract_ready_chunks(buffer, min_size, max_size)
            for chunk_text in chunks:
                clean_sentence = _clean_for_speech(chunk_text)
                if clean_sentence:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        path = f.name
                    before_tts = time.perf_counter()
                    await _synthesize(clean_sentence, path)
                    if first_chunk:
                        print(f"[time] first chunk synthesis (edge_tts): {time.perf_counter() - before_tts:.2f}s")
                        print(f"[time] total until first audio ready: {time.perf_counter() - start_total:.2f}s")
                    first_chunk = False
                    await queue.put((path, clean_sentence))
        print()
        clean_sentence = _clean_for_speech(buffer)
        if clean_sentence:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                path = f.name
            await _synthesize(clean_sentence, path)
            await queue.put((path, clean_sentence))
        await queue.put(None)

    producer_task = asyncio.create_task(producer())
    interrupted = False
    captured_audio = None
    first_audio = True
    spoken_text = []
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            path, clean_sentence = item
            if first_audio:
                _stop_loading_sound()
                _set_visual_state("falando")
                first_audio = False
            interrupted, captured_audio = await _play(path, mic_index, volume_threshold)
            os.remove(path)
            if not interrupted:
                spoken_text.append(clean_sentence)
            if interrupted:
                break
    finally:
        _stop_loading_sound()
        producer_task.cancel()
        try:
            await producer_task
        except (asyncio.CancelledError, Exception):
            pass

    return " ".join(spoken_text).strip(), interrupted, captured_audio


def respond_and_speak(history, mic_index, volume_threshold):
    return asyncio.run(_respond_and_speak(history, mic_index, volume_threshold))


def main():
    global _panel
    pygame.mixer.init()
    _panel = VisualPanel("Assistente de Voz")
    _panel.start()
    _load_whisper_model()

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0  # seconds of silence to consider the phrase finished
    recognizer.non_speaking_duration = 0.5

    mic_index = _find_microphone_index(MIC_NAME)
    if mic_index is None:
        print(f'Microphone "{MIC_NAME}" not found, using the system default.')
        microphone = sr.Microphone()
    else:
        name = sr.Microphone.list_microphone_names()[mic_index]
        print(f"Using microphone: [{mic_index}] {name}")
        microphone = sr.Microphone(device_index=mic_index)

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Calibrating ambient noise... please stay quiet for a moment.")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
    print(f"Calibrated energy threshold: {recognizer.energy_threshold:.0f}")
    interruption_volume_threshold = recognizer.energy_threshold * INTERRUPTION_VOLUME_MULTIPLIER

    config = _load_or_create_config(recognizer, microphone)
    assistant_name = config["nome_assistente"]
    _panel.set_agent_name(assistant_name)

    mode = "dormindo"
    pending_audio = None
    print(f'\nReady! Say "{assistant_name}" to activate. (Ctrl+C to exit)\n')

    while True:
        try:
            if pending_audio is not None:
                audio = pending_audio
                pending_audio = None
            else:
                _set_visual_state("ouvindo" if mode == "ativo" else "dormindo")
                with microphone as source:
                    print("Listening..." if mode == "ativo" else f'Waiting for "{assistant_name}"...')
                    audio = recognizer.listen(source, timeout=15, phrase_time_limit=20)

            print("Transcribing...")
            user_text = transcribe(audio)

            if not user_text:
                continue

            if mode == "dormindo":
                command = _extract_command_after_keyword(user_text, assistant_name)
                if command is None:
                    print(f'(heard: "{user_text}" - did not recognize the name "{assistant_name}")')
                    continue  # didn't say the name, ignore and keep sleeping

                mode = "ativo"
                if not command:
                    speak_short("Oi, pode falar.")
                    continue
                user_text = command
            elif _is_just_the_name(user_text, assistant_name):
                print(f"You: {user_text}")
                speak_short("Oi, pode falar.")
                continue

            print(f"You: {user_text}")

            if _contains_stop_phrase(user_text):
                speak_short("Ok, pode me chamar quando precisar.")
                mode = "dormindo"
                continue

            if _contains_change_voice_phrase(user_text):
                _choose_voice(recognizer, microphone)
                continue

            if _contains_change_name_phrase(user_text):
                assistant_name = _choose_name(recognizer, microphone)
                _panel.set_agent_name(assistant_name)
                print(f'Now responding as "{assistant_name}".')
                continue

            history.append({"role": "user", "content": user_text})

            response_text, interrupted, captured_audio = respond_and_speak(
                history, mic_index, interruption_volume_threshold
            )
            history.append({"role": "assistant", "content": response_text})
            if interrupted:
                print("(interrupted)")
                if captured_audio:
                    pending_audio = sr.AudioData(captured_audio, 16000, 2)

        except sr.UnknownValueError:
            continue
        except sr.WaitTimeoutError:
            continue
        except KeyboardInterrupt:
            print("\nShutting down.")
            break


if __name__ == "__main__":
    main()

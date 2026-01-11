# SPDX-License-Identifier: Apache-2.0
# PATCHED: Single-pass audio generation (no streaming/chunking)
print(">>> PATCHED serving_audio.py LOADED <<<", flush=True)
import base64
import io
import json
import os
import time
import traceback
from collections.abc import AsyncGenerator, AsyncIterator
from functools import lru_cache
from typing import Any, Final, Optional

import librosa
import numpy as np
from fastapi import Request
from pydub import AudioSegment
from starlette.datastructures import State

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         parse_chat_messages_futures,
                                         resolve_chat_template_content_format)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (AudioSpeechRequest,
                                              ChatCompletionMessageParam,
                                              RequestResponseMetadata)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.higgs_audio_tokenizer import AudioTokenizer
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .utils import create_audio_chunk

logger = init_logger(__name__)

OPENAI_TTS_SAMPLING_RATE = 24000
OPENAI_TTS_BIT_DEPTH = 16
OPENAI_TTS_CHANNELS = 1

TTS_SYSTEM_PROMPT = "Convert text to speech with the same voice."


@lru_cache(maxsize=50)
def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def pcm_to_target_format_bytes(pcm_data: np.ndarray, response_format: str,
                               original_sr: int, target_sr: int):
    """Convert PCM data to target format (wav, mp3, flac, pcm)."""
    audio_pcm16 = (pcm_data * np.iinfo(np.int16).max)\
                    .clip(np.iinfo(np.int16).min, np.iinfo(np.int16).max)\
                    .astype(np.int16)
    if response_format == "pcm":
        return audio_pcm16.tobytes()

    wav_audio = AudioSegment(
        audio_pcm16.tobytes(),
        frame_rate=original_sr,
        sample_width=OPENAI_TTS_BIT_DEPTH // 8,
        channels=OPENAI_TTS_CHANNELS,
    )
    if target_sr is not None and target_sr != original_sr:
        wav_audio = wav_audio.set_frame_rate(target_sr)

    target_io = io.BytesIO()
    wav_audio.export(target_io, format=response_format)
    target_io.seek(0)

    return target_io.getvalue()


def load_voice_presets(state: State,
                       voice_presets_dir: str,
                       interval: int = 10):
    while True:
        try:
            voice_file = os.path.join(voice_presets_dir, "config.json")
            if voice_file is not None:
                with open(voice_file) as f:
                    new_presents = json.load(f)
                diff = set(new_presents.keys()) - set(
                    state.voice_presets.keys())
                if len(diff) > 0:
                    logger.info("New voice presets added: %s", diff)
                state.voice_presets = new_presents
        except Exception as e:
            logger.error("Error loading voice presets: %s", str(e))
            logger.error("Detailed traceback:\n%s", traceback.format_exc())
            time.sleep(interval)


class HiggsAudioServingAudio(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        voice_presets_dir: str,
        chat_template_content_format: ChatTemplateContentFormatOption,
        *,
        request_logger: Optional[RequestLogger],
        audio_tokenizer: Optional[AudioTokenizer] = None,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)
        self.voice_presets_dir = voice_presets_dir
        self.request_logger = request_logger
        self.chat_template_content_format: Final = chat_template_content_format
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)

        self.audio_tokenizer = audio_tokenizer
        self.audio_num_codebooks = self.audio_tokenizer.num_codebooks
        self.audio_codebook_size = self.audio_tokenizer.codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate //
                                     self.audio_tokenizer_tps)
        self.audio_stream_bos_id = model_config.hf_config.audio_stream_bos_id
        self.audio_stream_eos_id = model_config.hf_config.audio_stream_eos_id

    def get_chat_template(self) -> str:
        return (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + "
            "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = bos_token + content %}"
            "{% endif %}"
            "{% if message['role'] == 'assistant' and '<|audio_bos|><|AUDIO|>' in message['content'] %}"
            "{% set content = content.replace('<|audio_bos|><|AUDIO|>', '<|audio_out_bos|><|AUDIO|>') %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>' }}"
            "{% endif %}")

    async def create_audio_speech_stream(
        self,
        request: AudioSpeechRequest,
        voice_presets: dict,
        raw_request: Optional[Request] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        PATCHED: Single-pass audio generation.
        Returns a generator that yields a single complete audio file.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        request_id = "audiospeech-" \
                     f"{self._base_request_id(raw_request)}"
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            sampling_params = request.to_sampling_params()
            self._log_inputs(request_id,
                             request.input,
                             params=sampling_params,
                             lora_request=None,
                             prompt_adapter_request=None)
            tokenizer = await self.engine_client.get_tokenizer(None)
            engine_prompt = await self.prepare_engine_prompt(
                request, tokenizer, voice_presets)
            generator = self.engine_client.generate(
                engine_prompt,
                sampling_params,
                request_id,
            )
            generators.append(generator)
        except ValueError as e:
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

        # PATCHED: Use single-pass generator instead of streaming
        print(">>> PATCHED: Returning single-pass generator <<<", flush=True)
        logger.info("PATCHED: Returning single-pass generator")
        return self.audio_speech_single_pass_generator(request, result_generator)

    async def prepare_engine_prompt(
            self,
            request: AudioSpeechRequest,
            tokenizer: AnyTokenizer,
            voice_presets: Optional[dict] = None) -> str:
        messages = self.prepare_messages(request, voice_presets)
        resolved_content_format = resolve_chat_template_content_format(
            self.get_chat_template(),
            None,
            self.chat_template_content_format,
            tokenizer,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        conversation, mm_data_future = parse_chat_messages_futures(
            messages,
            self.model_config,
            tokenizer,
            content_format=resolved_content_format,
        )

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=self.get_chat_template(),
            add_generation_prompt=True,
            continue_final_message=False,
            tools=None,
            documents=None,
        )

        request_prompt = apply_hf_chat_template(
            tokenizer,
            trust_remote_code=self.model_config.trust_remote_code,
            conversation=conversation,
            **_chat_template_kwargs,
        )

        mm_data = await mm_data_future
        prompt_inputs = await self._tokenize_prompt_input_async(
            request,
            tokenizer,
            request_prompt,
            truncate_prompt_tokens=None,
            add_special_tokens=False,
        )
        engine_prompt = TokensPrompt(
            prompt_token_ids=prompt_inputs["prompt_token_ids"])
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data

        return engine_prompt

    def prepare_messages(
        self,
        request: AudioSpeechRequest,
        voice_presets: Optional[dict] = None
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        audio_base64, reference_text, system_prompt = \
            self.tts_voice_raw(request.voice, self.voice_presets_dir, voice_presets)
        messages.append({
            "role": "system",
            "content": system_prompt or TTS_SYSTEM_PROMPT
        })

        # SFT model uses system prompt instead of reference audio for TTS
        if system_prompt is None:
            messages.append({"role": "user", "content": reference_text})
            messages.append({
                "role":
                "assistant",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    }
                }]
            })

        messages.append({"role": "user", "content": request.input})

        return messages

    def tts_voice_raw(self, voice: str, voice_presets_dir: str,
                      voice_presets: dict):
        if voice not in voice_presets:
            default_voice = list(voice_presets.keys())[0]
            logger.warning("Unsupported voice: %s, using default voice: %s",
                           voice, default_voice)
            voice = default_voice
        path = os.path.join(voice_presets_dir,
                            voice_presets[voice]["audio_file"])
        audio_base64 = encode_base64_content_from_file(path)
        return audio_base64, voice_presets[voice]["transcript"], voice_presets[
            voice].get("system_prompt", None)

    async def audio_speech_single_pass_generator(
        self,
        request: AudioSpeechRequest,
        result_generator: AsyncIterator[RequestOutput],
    ) -> AsyncGenerator[bytes, None]:
        """
        PATCHED: Single-pass audio generation.

        Instead of streaming chunks:
        1. Collect ALL audio tokens first
        2. Decode ALL tokens in one pass
        3. Return single WAV file with correct headers

        No chunking = no clicking. Single WAV = correct headers.
        """
        logger.info("PATCHED: Starting single-pass audio generation")
        start_time = time.time()

        # Accumulate ALL audio tokens
        audio_tokens_cache = np.ndarray((0, self.audio_num_codebooks), dtype=np.int64)

        try:
            iteration_count = 0
            async for res in result_generator:
                iteration_count += 1
                if iteration_count == 1:
                    logger.info("PATCHED: First iteration from result_generator")
                assert len(res.outputs) == 1, "Only one output should be generated per request"
                output = res.outputs[0]

                # Accumulate multimodal (audio) tokens
                if output.mm_token_ids is not None:
                    audio_tokens_cache = np.concatenate([
                        audio_tokens_cache,
                        output.mm_token_ids,
                    ], axis=0)
                    if iteration_count <= 5:
                        logger.info(f"PATCHED: Accumulated {audio_tokens_cache.shape[0]} tokens so far")

            gen_time = time.time() - start_time
            logger.info(f"Token generation: {audio_tokens_cache.shape[0]} tokens in {gen_time:.2f}s")

            if audio_tokens_cache.shape[0] == 0:
                logger.warning("No audio tokens generated")
                yield b''
                return

            # Decode ALL tokens in single pass
            decode_start = time.time()
            audio_chunk_size = audio_tokens_cache.shape[0]  # Use all tokens

            audio_waveform, _ = create_audio_chunk(
                audio_tokens_cache,
                audio_chunk_size,
                fade_out_audio=None,  # No previous chunk to fade from
                finalize=True,        # This is the final (and only) chunk
                audio_tokenizer=self.audio_tokenizer,
                audio_codebook_size=self.audio_codebook_size,
                samples_per_token=self.samples_per_token,
                audio_num_codebooks=self.audio_num_codebooks,
                audio_stream_bos_id=self.audio_stream_bos_id,
                audio_stream_eos_id=self.audio_stream_eos_id,
                return_as_numpy_audio=True,
            )

            if audio_waveform is None:
                logger.warning("Audio decoding returned None")
                yield b''
                return

            decode_time = time.time() - decode_start
            duration = len(audio_waveform) / self.audio_tokenizer.sampling_rate
            logger.info(f"Audio decode: {duration:.2f}s audio in {decode_time:.2f}s")

            # Resample if needed (24kHz is OpenAI TTS standard)
            if self.audio_tokenizer.sampling_rate != OPENAI_TTS_SAMPLING_RATE:
                audio_waveform = librosa.resample(
                    audio_waveform,
                    orig_sr=self.audio_tokenizer.sampling_rate,
                    target_sr=OPENAI_TTS_SAMPLING_RATE
                )

            # Convert to target format (single WAV with correct headers)
            output_bytes = pcm_to_target_format_bytes(
                audio_waveform,
                response_format=request.response_format,
                original_sr=OPENAI_TTS_SAMPLING_RATE,
                target_sr=OPENAI_TTS_SAMPLING_RATE,
            )

            total_time = time.time() - start_time
            logger.info(f"Total: {total_time:.2f}s for {duration:.2f}s audio")

            # Yield single complete audio file
            yield output_bytes

        except Exception as e:
            logger.exception("Error in single-pass audio generation")
            yield self.create_streaming_error_response(str(e))

        # Empty chunk signals end
        yield b''

    # Keep legacy streaming method for backwards compatibility if needed
    async def audio_speech_stream_generator(
        self,
        request: AudioSpeechRequest,
        result_generator: AsyncIterator[RequestOutput],
    ) -> AsyncGenerator[str, None]:
        """Legacy streaming generator - redirects to single-pass."""
        async for chunk in self.audio_speech_single_pass_generator(request, result_generator):
            yield chunk

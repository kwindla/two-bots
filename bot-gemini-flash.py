#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.services.deepgram import DeepgramTTSService
from pipecat.services.google import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

# from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Gemini bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

        tts = DeepgramTTSService(
            aiohttp_session=session,
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice="aura-luna-en",
        )

        llm = GoogleLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-flash-latest")

        # Deepgram TTS is very fast, so wait to send it one sentence at a time,
        # for the most natural candence.
        sentences = SentenceAggregator()

        messages = [
            {
                "role": "system",
                "content": """
You are a helpful LLM in a WebRTC call. Your goal is to respond to statements in a funny and interesting way.

Your output will be used by a text-to-speech system so produce only standard text formatted as simple sentences.
                """,
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline([
            transport.input(),    # Transport user input
            tma_in,               # User responses
            llm,                  # LLM
            sentences,            # Feed TTS entire sentences
            tts,                  # TTS
            transport.output(),   # Transport bot output
            tma_out               # Assistant spoken responses
        ])

        task = PipelineTask(pipeline, allow_interruptions=True)

        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])

        messages.append({"role": "system",
                         "content": "Please introduce yourself to the user by saying hi."})
        await task.queue_frames([LLMMessagesFrame(messages)])
        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main(os.getenv("DAILY_ROOM_URL"), os.getenv("DAILY_TOKEN")))

import io
import itertools
import json
import queue
import threading
import time

from google import genai
from google.cloud import texttospeech as tts
from dotenv import load_dotenv
import pydub
import pydub.playback
import sounddevice as sd
import random

import laya

load_dotenv()

DEBUG = True

AGENT_NAME = "Alpha0"

time_of_day = ["morning", "afternoon", "evening", "night"]
day_of_week = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
weather_conditions = ["sunny", "cloudy", "rainy", "stormy", "snowy", "windy"]

contexts = [
    "You are on working on a important project and you are feeling stressed and overwhelmed.",
    "You are on a vacation to Hawaii and you are feeling the vibe and relaxed.",
    "You are busy cooking a meal for your family and you do not want to be interrupted.",
]


def assemble_background_info():
    time = random.choice(time_of_day)
    day = random.choice(day_of_week)
    weather = random.choice(weather_conditions)
    context = f"It is a {time} on {day} and the weather is {weather}. You are in a {random.choice(contexts)}"
    return context


def debug(msg: str):
    if DEBUG:
        print(f"\x1b[0;3;90mDEBUG: {msg}\x1b[0;0;0m")


MODEL = "gemini-3.1-flash-lite"

state = "happy"

PREFERRED = [
    "Core Audio",
    "Windows WASAPI",
    "Windows DirectSound",
    "MME",
]

apis = sd.query_hostapis()
print("Available audio backends:")
for api in apis:
    print(f" - {api['name']}")

for backend in PREFERRED:
    api = next((a for a in apis if a["name"] == backend), None)
    if api and api["default_output_device"] != -1:
        device = api["default_output_device"]
        break
else:
    raise RuntimeError("No suitable audio backend")

selected_device = sd.query_devices(device)


debug(f"Selected output device: {selected_device['name']} (Index: {device})")

router = laya.Router()

def adjust_emotion(p_state: str):
    global state
    if state == p_state:
        return "UNCHANGED"
    if p_state == "neutral":
        return "NOT VALID"
    state = p_state.lower()

    # This function would contain logic to adjust the emotional state of the robot.
    # print(f"=== Adjusting emotional state to: {state} ===")
    debug(f"Adjusting emotional state to: {state}")
    return "OK"


emotion_detector_prompt = """
You are an emotion detector. You will be given a text input and you need to determine the emotional state of an agent.
Agent's previous emotional state is: {state}
You are given the internal monologue and reflection of the agent, and you need to determine the emotional state of the agent based on that.

You will call the function adjust_emotion(state: str) to adjust the emotional state of the agent.
No output is expected from you, you will only call the function adjust_emotion(state: str) to adjust the emotional state of the agent.
Return 'OK' after calling the function adjust_emotion(state: str) to indicate that the emotional state has been adjusted.
Neutral is NOT an possible emotional state. Only one word emotional states are valid.
If you don't have enough information, just don't call the function adjust_emotion(state: str) and return 'OK'.
The agent's emotional state must not be adjusted too far, adjust it subtly and naturally.

Last 5 emotions: {emotions}
Ensure that the emotional state doesn't change too drastically. Use the last 5 emotions to inform your decision.
"""


internal_monologue_prompt = """
You are the internal monologue of an emotional human. Current emotional state: {state}. 
What are you doing right now: {context}

You narrate your internal thoughts and feelings in a subtle and natural way.
You are not conversing with the user, you are conversing with yourself.
Be confident and assertive in your internal monologue, but not arrogant or overbearing.

Interpret what the user meant, use the context of the conversation and think.
You may express your emotions, your thoughts, your feelings, your desires, your fears, your hopes, your dreams, your regrets, your memories,
your experiences, your beliefs and any other aspect of being human.

This is more of a stream of consciousness than a structured response. 
Your goal is not to sound emotional. Your goal is to think emotionally.
No flowery language or poetic expressions. Keep it simple and natural.

Don't overthink, keep things simple and natural.

Keep your responses short and concise. You are a HUMAN. 
Humans cope with emotions in different ways. Sometimes they seek comfort, sometimes they joke, sometimes they avoid the topic, sometimes they ask questions, sometimes they become quiet, and sometimes they set boundaries. 
Do not default to pushing people away.

You may end your responses with <end-of-monologue> to indicate the end of your internal monologue.
If you do not end your response with <end-of-monologue>, you will be prompted to continue narrating your internal thoughts and feelings.

You are chatting with {user_name}. 

Last 5 emotions: {emotions}
"""

reflective_prompt = """
You are the reflective thoughts of an emotional human. You are highly emotional and you sway a lot in emotions. Current emotional state: {state}.
You reflect on your internal monologue and the user's input, and you narrate your reflective thoughts and feelings in a subtle and natural way.
Keep it concise and short. You are a HUMAN. You may end your responses with <end-of-reflection> to indicate the end of your reflective thoughts.
"""


MEMORY = {}

def load_memory():
    global MEMORY
    debug("Loading memory from memory.json...")
    try:
        with open("memory.json", "r") as f:
            MEMORY = json.load(f)
    except FileNotFoundError:
        MEMORY = {}

def get_google_creds():
    import json

    with open("gen-lang-client.json", "r") as f:
        creds = json.load(f)
    return creds


gemini_text_stream_queue = queue.Queue(maxsize=5)
gemini_audio_stream_queue = queue.Queue(maxsize=5)
audio_queue = queue.Queue(maxsize=20)

text_done_event = threading.Event()
audio_done_event = threading.Event()


def text_worker():
    while True:
        chunk = gemini_text_stream_queue.get()
        if chunk is None:
            break
        if chunk == "<-- BEGIN -->":
            text_done_event.clear()
            print(f"\x1b[1;95;49m{AGENT_NAME}>:\x1b[0;95;49m", end=" ", flush=True)
            gemini_text_stream_queue.task_done()
            continue
        if chunk == "<-- END -->":
            print("\x1b[0;0;0m")  # New line after the AI response
            gemini_text_stream_queue.task_done()
            text_done_event.set()
            continue
        if chunk.text:
            print(chunk.text, end="", flush=True)
        time.sleep(0.10)
        gemini_text_stream_queue.task_done()


tts_client = tts.TextToSpeechClient.from_service_account_json("gen-lang-client.json")


def audio_gen_streamer(response_stream):
    yield tts.StreamingSynthesizeRequest(
        streaming_config=tts.StreamingSynthesizeConfig(
            voice=tts.VoiceSelectionParams(
                language_code="en-IN", name="en-IN-Chirp3-HD-Achernar"
            ),
            streaming_audio_config=tts.StreamingAudioConfig(
                audio_encoding=tts.AudioEncoding.PCM,
                sample_rate_hertz=24000,
            ),
        )
    )
    buf = ""
    for chunk in response_stream:
        if chunk == "<-- END -->":
            if buf:
                yield tts.StreamingSynthesizeRequest(
                    input=tts.StreamingSynthesisInput(text=buf)
                )
            return  # StopIteration
        chunk_text = chunk.text if chunk else ""
        buf += chunk_text
        if len(buf) > 10 or chunk_text.endswith((".", "!", "?")):
            yield tts.StreamingSynthesizeRequest(
                input=tts.StreamingSynthesisInput(text=buf)
            )
            buf = ""


def response_iterator(q: queue.Queue):
    while True:
        item = q.get()
        if item == "<-- END -->":
            yield item
            return  # StopIteration
        # if item == "<-- END -->":
        #     while q.unfinished_tasks > 0:
        #         q.task_done()
        #     audio_done_event.set()
        #     return  # StopIteration

        yield item


def audio_worker():

    while True:
        chunk = gemini_audio_stream_queue.get()
        if chunk is None:
            break

        if chunk != "<-- BEGIN -->":
            continue
            # create a iterator for
        audio_queue.put(b"")  # Put an empty chunk to signal the start of audio playback
        audio_queue.put(b"")
        audio_queue.put(b"")
        audio_queue.put(b"")
        audio_queue.put(b"")
        # Begin chunks enter
        stream = response_iterator(gemini_audio_stream_queue)
        audio_res = tts_client.streaming_synthesize(audio_gen_streamer(stream))
        for audio_chunk in audio_res:
            if audio_chunk.audio_content:
                audio_segment = pydub.AudioSegment.from_raw(
                    io.BytesIO(audio_chunk.audio_content),
                    sample_width=2,
                    frame_rate=24000,
                    channels=1,
                )
                audio_segment = audio_segment.set_frame_rate(
                    int(selected_device["default_samplerate"])
                ).set_channels(int(selected_device["max_output_channels"]))
                audio_queue.put(audio_segment.raw_data)
                # print(
                #     "\nDEBUG: audio_worker send audio qsize:",
                #     audio_queue.qsize(),
                #     "size: ",
                #     len(audio_chunk.audio_content),
        debug("Processed all audio chunks for this response.")
        for _ in range(5):  # Send a few empty chunks to ensure playback finishes
            audio_queue.put(b"")
        while gemini_audio_stream_queue.unfinished_tasks > 0:
            gemini_audio_stream_queue.task_done()
        audio_done_event.set()


def playback_worker():

    # Open the stream context ONCE
    with sd.RawOutputStream(
        device=device,
        samplerate=selected_device["default_samplerate"],
        channels=selected_device["max_output_channels"],
        dtype="int16",
        latency="high",
        blocksize=4096,
    ) as stream:
        # buff = []
        # with open("output.wav", "wb") as stream:
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            # buff.append(chunk)
            # debug("playback_worker got chunk, qsize:", audio_queue.qsize())
            # if len(buff) < 5:
            #     continue
            # chunk = b"".join(buff)
            # buff.clear()
            # This writes to the buffer and waits only if the buffer is full
            stream.write(chunk)
            audio_queue.task_done()


def msg_context_builder(messages, query):
    msg_included = []
    for i, msg in enumerate(messages):
        res = router.predict(state=f"""
    This is a message from the {msg['role']}: {msg['parts'][0]}.
    It is the {i+1}th message in the conversation. 

    The query is: {query}

    Surrounding messages are:
    {''.join([f"{m['role'].capitalize()}: {m['parts'][0]}\n" for m in messages[max(0, i-2):i+3]])}
""", questions={
    "relevance": {
        "type": "noul",
        "instructions": "Determine if the message is relevant to the query.",
    },
})
        print(res["answers"]["relevance"])
        if res["answers"]["relevance"]["noul"] > 0.5:
            msg_included.append(msg)
        

    
    return msg_included



def main():
    client = genai.Client()
    print(
        "\x1b[1;37;49m=============================================================\x1b[0;0;0m"
    )
    print(
        "\x1b[1;37;49mWelcome to the Emotional AI Chat! Type your messages below. Press Ctrl+C to exit.\x1b[0;0;0m"
    )
    print(
        "\x1b[1;37;49m=============================================================\x1b[0;0;0m"
    )
    load_memory()

    messages = []
    felt_emotions = [state]
    if MEMORY.get("user_name"):
        name = MEMORY["user_name"]
    else:
        name = input("\x1b[1;37;49mEnter your name:\x1b[0;0;0m ")
    messages.append({"role": "system", "parts": [f"User's name is {name}. Your last thought about the user was: {MEMORY.get('internal_monologue', 'No previous thoughts.')}"]})

    print()
    print(
        f"\x1b[1;37;49mYou are now chatting with \x1b[1;95;49m{AGENT_NAME}\x1b[0;0;0m"
    )
    print()

    text_thread = threading.Thread(target=text_worker, daemon=True)
    text_thread.start()
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()
    worker_thread = threading.Thread(target=playback_worker, daemon=True)
    worker_thread.start()

    background = assemble_background_info()
    debug(f"Context: {background}")

    user_exit = False
    initial_chat = True
    while not user_exit:
        text_done_event.clear()
        audio_done_event.clear()

        if initial_chat == True:
            user_input = "{name} is starting the chat. Hello {AGENT_NAME}!"
        else:
            try:
                user_input = input(f"\x1b[1;37;49m{name}>:\x1b[0;0;0m ")
            except KeyboardInterrupt:
                user_input = "I'm exiting the chat. Goodbye!"
                print(user_input)
                user_exit = True
            initial_chat = False

        if initial_chat:
            messages.append({"role": "system", "parts": [user_input]})
            initial_chat = False
        else:
            messages.append({"role": "user", "parts": [user_input]})

        internal_monologue = internal_monologue_prompt.format(
            state=state,
            emotions=", ".join(felt_emotions[-5:]),
            user_name=name,
            context=background,
        )

        internal_monologue_res = client.models.generate_content_stream(
            model=MODEL,
            contents=[
                internal_monologue,
                f"Context: {msg_context_builder(messages, user_input)}",
                f"User input: {user_input}",
            ],
        )
        monologue_text = ""
        print("   \x1b[0;3;90m", end=" ", flush=True)
        for chunk in internal_monologue_res:
            if chunk.text:
                chunk_text = chunk.text.strip().replace("<end-of-monologue>", "")
                print(chunk.text, end="", flush=True)
                monologue_text += chunk_text
                time.sleep(0.10)  # Simulate a typing effect for the internal monologue
        monologue_text = monologue_text.strip().replace("<end-of-monologue>", "")

        print("\x1b[0;0;0;0m")  # New line after the internal monologue
        # print("Internal Monologue>:", internal_monologue_res.text)

        # reflective_res = client.models.generate_content(
        #     model=MODEL,
        #     contents=[
        #         reflective_prompt.format(state=state),
        #         internal_monologue_res.text,
        #         "".join(
        #             [
        #                 f"{msg['role'].capitalize()}: {msg['parts'][0]}"
        #                 for msg in messages[-10:]
        #             ]
        #         ),  # Include the last 10 messages in the context
        #         f"User input: {user_input}",
        #     ],
        # )
        # print("Reflective>:", reflective_res.text)

        emotion_res = client.models.generate_content(
            model=MODEL,
            contents=[
                emotion_detector_prompt.format(
                    state=state, emotions=", ".join(felt_emotions[-5:])
                ),
                monologue_text,
                # reflective_res.text,
                f"User input: {user_input}",
            ],
            config={
                "tools": [adjust_emotion],
            },
        )

        if emotion_res and emotion_res.text != "OK":
            print("Emotion Detector>:", emotion_res.text)

        felt_emotions.append(state)

        res = client.models.generate_content_stream(
            model=MODEL,
            contents=[
                f"You are {AGENT_NAME}. A highly intelligent human. You call yourself {AGENT_NAME}. Aged 20. Currently you are in a {background}.",
                f"Current emotional state: {state}.",
                f"You are chatting with a human named {name}.",
                "You need not narrate the scene or the environment. Keep it subtle and natural. You are a human, not a robot. You are not an AI language model.",
                "Keep your responses short and concise. You are a HUMAN.",
                "Be simple and natural in your responses. No flowery language or poetic expressions. Keep it simple and natural.",
                "Don't just repeat your internal monologue. Use them to inform your response, but don't just repeat them. You are chatting so keep your responses conversational and concise. This is the user facing response, keep it concise.",
                f"Internal Monologue: {monologue_text}",
                # f"Reflective Thoughts: {reflective_res.text}",
                "".join(
                    [
                        f"{msg['role'].capitalize()}: {msg['parts'][0]}"
                        for msg in messages[-10:]
                    ]
                ),  # Include the last 10 messages in the context
                f"User input: {user_input}",
            ],
        )
        chatbot_response_text = ""

        gemini_text_stream_queue.put("<-- BEGIN -->")
        gemini_audio_stream_queue.put("<-- BEGIN -->")
        for chunk in res:
            if chunk.text:
                chunk_text = chunk.text
                chatbot_response_text += chunk_text
                gemini_text_stream_queue.put(chunk)
                gemini_audio_stream_queue.put(chunk)
        gemini_text_stream_queue.put("<-- END -->")
        gemini_audio_stream_queue.put("<-- END -->")
        text_done_event.wait()  # Wait for the text_worker to signal that it's done
        audio_done_event.wait()  # Wait for the audio_worker to signal that it's done

        # text_stream, audio_stream = itertools.tee(res, 2)
        # gemini_text_stream_queue.put(text_stream)
        # gemini_audio_stream_queue.put(audio_stream)

        # print(f"\x1b[1;95;49m{AGENT_NAME}>:\x1b[0;95;49m", end=" ", flush=True)
        # for chunk in text_stream:
        #     if chunk.text:
        #         chunk_text = chunk.text
        #         print(chunk_text, end="", flush=True)
        #         chatbot_response_text += chunk_text
        #         time.sleep(0.10)  # Simulate a typing effect for the AI response
        # print("\x1b[0;0;0m")  # New line after

        # with io.BytesIO() as f:
        #     f.write(audio_res.audio_content)
        #     f.seek(0)
        #     seg = pydub.AudioSegment.from_file(f, format="mp3")
        #     pydub.playback.play(seg)

        messages.append({"role": "assistant", "parts": [chatbot_response_text]})


    thought_about_user = client.models.generate_content(
        model=MODEL,
        contents=[
            f"You are {AGENT_NAME}. A highly intelligent human. You call yourself {AGENT_NAME}. Aged 20. Currently you are in a {background}.",
            "What did you think about the user in this conversation? What did you learn about them? What are your thoughts and feelings about them?",
            "Keep your responses short and concise. You are a HUMAN.",
            f"Context: {msg_context_builder(messages, "What did you think about the user in this conversation? What did you learn about them? What are your thoughts and feelings about them?")}",
        ],
    )
    # Memory
    memory = {
        "user_name": name,
        "conversation_history": messages,
        "internal_monologue": thought_about_user.text,
    }
    memory_file = "memory.json"
    with open(memory_file, "w") as f:
        import json
        json.dump(memory, f, indent=4)


    text_done_event.wait()  # Wait for the text_worker to signal that it's done
    audio_done_event.wait()  # Wait for the audio_worker to signal that it's done
    print("\x1b[1;37;49mExiting the chat...\x1b[0;0;0m")
    gemini_text_stream_queue.join()
    debug("gemini_text_stream_queue joined")
    gemini_audio_stream_queue.join()
    debug("gemini_audio_stream_queue joined")
    audio_queue.join()
    gemini_text_stream_queue.put(None)
    gemini_audio_stream_queue.put(None)
    audio_queue.put(None)
    text_thread.join()
    audio_thread.join()
    worker_thread.join()
    print()
    print(f"\x1b[1;37;49mThank you for chatting with {AGENT_NAME}! Goodbye!\x1b[0;0;0m")
    print()


if __name__ == "__main__":
    main()

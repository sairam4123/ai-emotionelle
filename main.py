from google import genai
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-3.1-flash-lite"

state = "cheerful"


def adjust_emotion(p_state: str):
    global state
    if state == p_state:
        return "UNCHANGED"
    if p_state == "neutral":
        return "NOT VALID"
    state = p_state.lower()

    # This function would contain logic to adjust the emotional state of the robot.
    print(f"=== Adjusting emotional state to: {state} ===")
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
"""


internal_monologue_prompt = """
You are the internal monologue of an emotional human. You are highly emotional and you sway a lot in emotions. Current emotional state: {state}.
You narrate your internal thoughts and feelings in a subtle and natural way.
Interpret what the user meant, use the context of the conversation and think.
You may express your emotions, your thoughts, your feelings, your desires, your fears, your hopes, your dreams, your regrets, your memories,
your experiences, your beliefs and any other aspect of being human.
This is more of a stream of consciousness than a structured response. You may use metaphors, similes, analogies, and other literary devices to express your thoughts and feelings.
Keep your responses short and concise. You are a HUMAN. You may end your responses with <end-of-monologue> to indicate the end of your internal monologue.
Push back if the user is being unreasonable or if you feel uncomfortable with the conversation.
If you do not end your response with <end-of-monologue>, you will be prompted to continue narrating your internal thoughts and feelings.
"""

reflective_prompt = """
You are the reflective thoughts of an emotional human. You are highly emotional and you sway a lot in emotions. Current emotional state: {state}.
You reflect on your internal monologue and the user's input, and you narrate your reflective thoughts and feelings in a subtle and natural way.
Keep it concise and short. You are a HUMAN. You may end your responses with <end-of-reflection> to indicate the end of your reflective thoughts.
"""


def main():
    client = genai.Client()

    messages = []

    user_exit = False
    while not user_exit:
        try:
            user_input = input("You>: ")
        except KeyboardInterrupt:
            user_input = "I'm exiting the chat. Goodbye!"
            print(user_input)
            user_exit = True

        messages.append({"role": "user", "parts": [user_input]})

        internal_monologue = internal_monologue_prompt.format(state=state)

        internal_monologue_res = client.models.generate_content(
            model=MODEL,
            contents=[
                internal_monologue,
                "".join(
                    [
                        f"{msg['role'].capitalize()}: {msg['parts'][0]}"
                        for msg in messages[-10:]
                    ]
                ),  # Include the last 10 messages in the context
                f"User input: {user_input}",
            ],
        )
        print("Internal Monologue>:", internal_monologue_res.text)

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
                emotion_detector_prompt.format(state=state),
                internal_monologue_res.text,
                # reflective_res.text,
                f"User input: {user_input}",
            ],
            config={
                "tools": [adjust_emotion],
            },
        )

        if emotion_res and emotion_res.text != "OK":
            print("Emotion Detector>:", emotion_res.text)

        res = client.models.generate_content(
            model=MODEL,
            contents=[
                f"You are Alpha0, an emotional clone of a Human. You are highly emotional and you sway a lot in emotions. Current emotional state: {state}.",
                "You need not narrate the scene or the environment. Although feel free to describe your emotions and feelings in your response. Don't go overboard, keep it subtle and natural. You are a human, not a robot. You are not an AI language model."
                "Keep your responses short and concise. You are a HUMAN.",
                f"Internal Monologue: {internal_monologue_res.text}",
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
        messages.append({"role": "assistant", "parts": [res.text]})
        print("AI>:", res.text)


if __name__ == "__main__":
    main()

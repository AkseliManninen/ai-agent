import os
import gradio as gr
import requests
import inspect
import pandas as pd

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
import dotenv

from faster_whisper import WhisperModel
from PIL import Image
from prompts import INSTRUCTIONS
from prompts import SYSTEM_MESSAGE
from smolagents import CodeAgent
from smolagents import DuckDuckGoSearchTool
from smolagents import LiteLLMModel
from smolagents import OpenAIServerModel
from smolagents.tools import tool
from smolagents import ToolCallingAgent
from smolagents import WikipediaSearchTool
from youtube_transcript_api import YouTubeTranscriptApi

dotenv.load_dotenv()


@tool
def read_python_file(file_path: str) -> str:
    """
    Read and return the contents of a local .py file.
    Args:
        file_path: str path to the .py file
    """
    if not file_path.endswith(".py"):
        return "[ERROR] Only .py files are allowed."

    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"[ERROR] Failed to read file: {e}"


@tool
def transcribe_video(video_or_path: str) -> str:
    """Transcribe a youtube video.
    Args:
        video_or_path: path to video str
    """
    if video_or_path.lower().startswith("http"):
        vid = video_or_path.split("v=")[-1].split("&")[0]
        txt = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        return " ".join([seg["text"] for seg in txt])
    else:
        return "The input was not valid."


@tool
def reverse_text(text: str) -> str:
    """Reverse a string that starts with a dot.
    Args:
        text: the original question str
    """
    if text.startswith("."):
        reversed_text = text[1:][::-1]
        return reversed_text
    return text


wikipedia = WikipediaSearchTool()


class AkiAgent:
    def __init__(self):
        print("Starting AkiAgent.")

        model = LiteLLMModel(
            model_id="azure/gpt-4.1-mini",
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            flatten_messages_as_text=True,
        )

        self.agent = CodeAgent(
            model=model,
            tools=[read_python_file, reverse_text, transcribe_video, wikipedia],
            add_base_tools=True,
            executor_type=None,
            max_steps=10,
            additional_authorized_imports=[
                "csv",
                "faster_whisper",
                "markdown",
                "openpyxl",
                "pandas",
                "python_chess",
                "requests",
                "stockfish",
                "youtube_transcript_api",
            ],
        )

        # Use the GAIA system prompt.
        self.agent.system_prompt = self.agent.system_prompt + SYSTEM_MESSAGE

    def __call__(self, question: str, file_name: str) -> str:
        print(f"Agent received question: {question[:60]}...")
        prompt = reverse_text(question) + INSTRUCTIONS + file_name

        try:
            final_answer = self.agent.run(prompt)
        except Exception as e:
            final_answer = f"[ERROR] Agent failed: {e}"
        print(f"Returning: {final_answer}")
        return final_answer


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = AkiAgent()

    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        print(f"Processing the next item: {item}")
        task_id = item.get("task_id")
        question_text = item.get("question")
        file = item.get("file_name")
        print(f"Received a file: {file}")
        if file:
            try:
                file_response = requests.get(f"{api_url}/files/{task_id}", timeout=15)
                if file_response.status_code == 200:
                    file_content = file_response.content
                    # Save to file, using a safe filename.
                    filename = item.get("file_name") or f"{task_id}_file"
                    file_path = os.path.join(".", filename)
                    with open(file_path, "wb") as f:
                        f.write(file_content)
                    print(f"Saved file for task {task_id} to {file_path}")
                elif file_response.status_code == 404:
                    print(f"No file found for task {task_id}")
                else:
                    print(
                        f"Failed to fetch file for task {task_id}: Status {file_response.status_code}"
                    )
            except Exception as e:
                print(f"Error fetching file for task {task_id}: {e}")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text, file)
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted_answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": submitted_answer,
                }
            )
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": f"AGENT ERROR: {e}",
                }
            )

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main"
        )
    else:
        print(
            "ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined."
        )

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)

import gradio as gr
import requests

API_URL = "http://localhost:9009/v1/chat/completions"


def chat_with_model(
    message,
    history,
    temperature,
    max_tokens,
    use_cv,
    cv_name,
    cv_path,
    cv_scale,
    cv_degree,
    cv_keep_norm,
):
    messages = []
    for user_msg, _ in history:
        messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "user", "content": message})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if use_cv:
        payload["control_vector"] = {
            "name": cv_name,
            "path": cv_path,
            "scale": cv_scale,
            "target_degree": cv_degree,
            "keep_norm": cv_keep_norm,
        }

    try:
        resp = requests.post(API_URL, json=payload)
        resp.raise_for_status()  # Raise error for bad status codes

        print(f"Server response status: {resp.status_code}")
        print(f"Response content: {resp.text}")

        return resp.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return f"Error communicating with server: {str(e)}"
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Response parsing error: {e}")
        print(f"Response content: {resp.text}")
        return f"Error parsing server response: {str(e)}"


def compute_cv_name(path, degree):
    # Extract filename from path without extension
    import os

    filename = os.path.splitext(os.path.basename(path))[0]
    return f"{filename}-{degree}"


def create_ui():
    with gr.Blocks() as demo:
        gr.HTML("<h1>LLM Chat with Control Vectors</h1>")

        with gr.Row():
            # Left column for settings
            with gr.Column(scale=1):
                with gr.Group():
                    temperature = gr.Slider(
                        minimum=0, maximum=2, value=0, label="Temperature"
                    )
                    max_tokens = gr.Slider(
                        minimum=1, maximum=2048, value=256, label="Max Tokens"
                    )
                    use_cv = gr.Checkbox(label="Use Control Vector")
                    with gr.Column(visible=False) as cv_settings:
                        cv_name = gr.Textbox(
                            label="Control Vector Name", value="", interactive=False
                        )
                        cv_path = gr.Textbox(
                            label="Vector Path", value="/path/to/vector.npy"
                        )
                        cv_scale = gr.Slider(
                            label="Scale Factor", minimum=-10, maximum=10, value=0
                        )
                        cv_degree = gr.Slider(
                            label="Target Degree",
                            minimum=0,
                            maximum=360,
                            step=10,
                            value=0,
                        )
                        cv_keep_norm = gr.Checkbox(label="Keep Norm")

                def toggle_cv_settings(use_cv_flag):
                    return gr.update(visible=use_cv_flag)

                use_cv.change(toggle_cv_settings, use_cv, cv_settings)

                def update_cv_name(path, degree):
                    return compute_cv_name(path, degree)

                cv_path.change(
                    update_cv_name, inputs=[cv_path, cv_degree], outputs=[cv_name]
                )
                cv_degree.release(
                    update_cv_name, inputs=[cv_path, cv_degree], outputs=[cv_name]
                )

            # Right column for chat
            with gr.Column(scale=2, min_width=800):
                chatbot = gr.Chatbot(placeholder="What's on your mind ?", height=800)
                chat = gr.ChatInterface(
                    chatbot=chatbot,
                    fn=chat_with_model,
                    additional_inputs=[
                        temperature,
                        max_tokens,
                        use_cv,
                        cv_name,
                        cv_path,
                        cv_scale,
                        cv_degree,
                        cv_keep_norm,
                    ],
                )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()

import time
import gradio as gr
from ragChain import submitUserMessage

def chat(message:str, history):
    return submitUserMessage(message)


def slow_echo_chat(message, history):
    answer = submitUserMessage(message)
    for i in range(len(answer)):
        time.sleep(0.01)
        yield answer[: i+1]
        
        
# gr.ChatInterface(chat).launch()
interface = gr.ChatInterface(slow_echo_chat)

interface.launch()

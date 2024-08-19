import time
import gradio as gr
from ragChain import submitUserMessage

def chat(message:str, history):
    return submitUserMessage(message)


def slow_echo_chat(message, history):
    for i in range(len(submitUserMessage(message))):
        time.sleep(0.2)
        yield message[: i+1]
        
        
# gr.ChatInterface(chat).launch()
gr.ChatInterface(slow_echo_chat).launch()

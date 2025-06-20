import gradio as gr

class Parameter:
    def __init__(self, default, choices=None) -> None:
        self.default = default
        self.choices = choices

class IntParameter(Parameter):
    pass

class StrParameter(Parameter):
    pass
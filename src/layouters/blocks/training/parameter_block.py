import gradio as gr

from src.layouters.blocks.layout_block import LayoutBlock
from .utils import argument_tuples_to_components


class ArgumentBasedParameterBlock(LayoutBlock):
    title = ""
    arg_handler = None
    arguments = None

    def __init__(self, tracker_prefix=""):
        super().__init__(tracker_prefix)

    def layout(self, tracker, root):
        with gr.Accordion(self.title):
            argument_tuples_to_components(
                arg_handler=self.arg_handler,
                arguments=self.arguments,
                tracker=tracker,
                tracker_prefix=self.tracker_prefix,
            )

# class CompoundBlock(ArgumentBasedParameterBlock):

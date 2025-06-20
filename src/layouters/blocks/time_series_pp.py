import gradio as gr

from . import LayoutBlock


class TimeSeriesPostProcessingParameterBlock(LayoutBlock):
    def layout(self, tracker, root):
        with gr.Row():
            with gr.Column():
                tracker["tspp_type"] = gr.Dropdown([
                    "None",
                    "Moving Average - Constant Kernel",
                    "Moving Average - Exponential Kernel",
                    "Moving Average - Linear Kernel",
                    "Moving Average - Gaussian Kernel",
                ], value="None", label="Time Series Post-Processing")
            with gr.Column():
                tracker["tspp_kernel_size"] = gr.Number(15, label="Kernel size", minimum=1, precision=0, step=2)
                tracker["tspp_kernel_base"] = gr.Number(0.8, label="Kernel exponential base", minimum=0, maximum=1, step=0.05, precision=1)
                tracker["tspp_kernel_sigma"] = gr.Number(15, label="Gaussian kernel sigma", step=0.5, precision=1, minimum=0.5)

            def tspp_change(tspp_type, tspp_kernel_size, tspp_kernel_base, tspp_kernel_sigma):
                if tspp_type == "None":
                    tspp_kernel_size = gr.Number(visible=False)
                    tspp_kernel_base = gr.Number(visible=False)
                    tspp_kernel_sigma = gr.Number(visible=False)
                elif tspp_type == "Moving Average - Constant Kernel":
                    tspp_kernel_size = gr.Number(visible=True)
                    tspp_kernel_base = gr.Number(visible=False)
                    tspp_kernel_sigma = gr.Number(visible=False)
                elif tspp_type == "Moving Average - Exponential Kernel":
                    tspp_kernel_size = gr.Number(visible=True)
                    tspp_kernel_base = gr.Number(visible=True)
                    tspp_kernel_sigma = gr.Number(visible=False)
                elif tspp_type == "Moving Average - Linear Kernel":
                    tspp_kernel_size = gr.Number(visible=True)
                    tspp_kernel_base = gr.Number(visible=False)
                    tspp_kernel_sigma = gr.Number(visible=False)
                elif tspp_type == "Moving Average - Gaussian Kernel":
                    tspp_kernel_size = gr.Number(visible=True)
                    tspp_kernel_base = gr.Number(visible=False)
                    tspp_kernel_sigma = gr.Number(visible=True)

                return tspp_kernel_size, tspp_kernel_base, tspp_kernel_sigma

            tracker["tspp_type"].change(
                tspp_change,
                inputs=[
                    tracker["tspp_type"],
                    tracker["tspp_kernel_size"],
                    tracker["tspp_kernel_base"],
                    tracker["tspp_kernel_sigma"],
                ],
                outputs=[
                    tracker["tspp_kernel_size"],
                    tracker["tspp_kernel_base"],
                    tracker["tspp_kernel_sigma"],
                ],
            )
            root.load(
                tspp_change,
                inputs=[
                    tracker["tspp_type"],
                    tracker["tspp_kernel_size"],
                    tracker["tspp_kernel_base"],
                    tracker["tspp_kernel_sigma"],
                ],
                outputs=[
                    tracker["tspp_kernel_size"],
                    tracker["tspp_kernel_base"],
                    tracker["tspp_kernel_sigma"],
                ],
            )
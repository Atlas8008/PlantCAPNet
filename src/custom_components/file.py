import gradio as gr


# Sorted downloaded files
class File(gr.File):
    def _download_files(self, value):
        # print("File download")
        # print(value)
        # if isinstance(value, list):
        #     value = sorted(value)
        # print(value)

        return super()._download_files(value)

    # def preprocess(self, payload: ListFiles | gr.FileData | None) -> bytes | str | list[bytes] | list[str] | None:
    #     print("File preprocess")
    #     return super().preprocess(payload)

    def postprocess(self, value):
        print("File postprocess")
        return super().postprocess(value)
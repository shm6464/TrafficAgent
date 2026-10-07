"""Headless（无 UI）的 FileIndex UI 占位类。

用于在脚本化入库 / 检索时绕开 `ktem.index.file.ui`（该模块顶部 `import gradio`，
而 gradio 4.x 依赖已在新版 huggingface_hub 中移除的 `HfFolder`，会导致导入失败）。

本模块提供与 `ui.FileIndexPage` / `ui.FileSelector` 同名的占位类，仅需满足
`FileIndex._setup_file_index_ui_cls()` / `_setup_file_selector_ui_cls()` 的
`import_dotted_string` 导入即可，不实现任何 UI 逻辑（入库链路 `get_indexing_pipeline`
完全不使用 UI 类）。

用法：在 `flowsettings.py` 或索引 `config` 中注入
    FILE_INDEX_UI = "ktem.index.file.headless_ui.FileIndexPage"
    FILE_INDEX_SELECTOR_UI = "ktem.index.file.headless_ui.FileSelector"
"""


class FileSelector:
    """无 UI 依赖的文件选择器占位类。"""

    def __init__(self, app, index):
        self._app = app
        self._index = index

    def get_selected_ids(self, selected):
        return selected


class FileIndexPage:
    """无 UI 依赖的索引页占位类。"""

    def __init__(self, app, index):
        self._app = app
        self._index = index

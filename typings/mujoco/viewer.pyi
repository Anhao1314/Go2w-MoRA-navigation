"""mujoco.viewer 子模块类型桩。"""

from typing import Any, Callable, Optional

KeyCallbackType = Callable[[int], None]


class Handle:
    def is_running(self) -> bool: ...
    def sync(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Handle: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...


def launch_passive(
    model: Any,
    data: Any,
    *,
    key_callback: Optional[KeyCallbackType] = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> Handle: ...

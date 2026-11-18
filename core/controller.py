import time
import pydirectinput

# Disable pydirectinput's default 0.1s sleep between calls so 20 FPS loops stay fast
pydirectinput.PAUSE = 0.0

DEFAULT_KEYMAP = {
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d",
    "jump": "j",
    "action": "k",
    "cam_left": "q",
    "cam_right": "e",
}


class SA2Controller:
    """DirectInput keyboard controller interface for Sonic Adventure 2."""

    def __init__(
        self,
        keymap: dict[str, str] | None = None,
        deadzone: float = 0.3,
        backend=None,
    ):
        self.keymap = {**DEFAULT_KEYMAP, **(keymap or {})}
        self.deadzone = deadzone
        self.backend = backend if backend is not None else pydirectinput
        self._held_keys: set[str] = set()

    def set_state(
        self,
        lx: float = 0.0,
        ly: float = 0.0,
        jump: bool = False,
        action: bool = False,
        cam_left: bool = False,
        cam_right: bool = False,
    ) -> None:
        """
        Applies a single frame of inputs via DirectInput scan codes.
        - lx, ly: Movement axes in [-1.0, 1.0] (ly > deadzone presses 'up')
        - jump / action / cam_left / cam_right: Boolean button states
        """
        desired: set[str] = set()

        if ly > self.deadzone:
            desired.add(self.keymap["up"])
        elif ly < -self.deadzone:
            desired.add(self.keymap["down"])

        if lx > self.deadzone:
            desired.add(self.keymap["right"])
        elif lx < -self.deadzone:
            desired.add(self.keymap["left"])

        if jump:
            desired.add(self.keymap["jump"])
        if action:
            desired.add(self.keymap["action"])
        if cam_left:
            desired.add(self.keymap["cam_left"])
        if cam_right:
            desired.add(self.keymap["cam_right"])

        # Release keys no longer active, then press newly active keys
        for key in self._held_keys - desired:
            self.backend.keyUp(key)
        for key in desired - self._held_keys:
            self.backend.keyDown(key)

        self._held_keys = desired

    def reset(self) -> None:
        """Releases all currently held keys."""
        for key in list(self._held_keys):
            self.backend.keyUp(key)
        self._held_keys.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()


if __name__ == "__main__":
    print("Starting DirectInput keyboard test...")
    with SA2Controller() as ctrl:
        print("Click into the Sonic Adventure 2 window! Starting in 5 seconds...")
        for i in range(5, 0, -1):
            print(f"  {i}...")
            time.sleep(1.0)

        print("Action: Run forward (1.5s)")
        ctrl.set_state(ly=1.0)
        time.sleep(1.5)

        print("Action: Jump while running (0.3s)")
        ctrl.set_state(ly=1.0, jump=True)
        time.sleep(0.3)

        print("Action: Release jump, keep running (0.7s)")
        ctrl.set_state(ly=1.0, jump=False)
        time.sleep(0.7)

        print("Action: Action/Spin Dash tap (0.3s)")
        ctrl.set_state(action=True)
        time.sleep(0.3)

        print("Test complete. All keys released.")
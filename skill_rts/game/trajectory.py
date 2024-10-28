from skill_rts.game.game_state import GameState
import json

ACTION = {0: "noop", 1: "move", 2: "harvest", 3: "return", 4: "produce", 5: "attack"}
DIRECTION = {0: "north", 1: "east", 3: "south", 4: "west"}


class Trajectory:
    def __init__(self, raw_traj):
        self.raw_traj = raw_traj
    
    def get_gametime(self):
        """Get the latest game time."""
        return self.raw_traj["entries"][-1]["time"]
    
    def get_game_state(self, gametime=None) -> GameState:
        """Get the game state at the specified time."""
        if gametime is None:
            entry = self.raw_traj["entries"][-1]
        else:
            entry = next((e for e in self.raw_traj["entries"] if e["time"] == gametime), None)
        return GameState(entry) if entry else None
    
    def to_string(self) -> str:
        text = ""
        for entry in self.raw_traj["entries"]:
            gs = GameState(entry)
            text += gs.to_string() + "\n"
        return text

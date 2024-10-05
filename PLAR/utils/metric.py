RESOURCES_COST_MAP = {
    "worker": 1,
    "light": 2,
    "heavy": 2,
    "ranged": 2,
    "base": 10,
    "barrack": 5
}

UNIT_HP_MAP = {
    "worker": 1,
    "light": 4,
    "heavy": 4,
    "ranged": 1,
    "base": 10,
    "barrack": 4
}

class Metric:

    def __init__(self, obs_dict):
        self.metric = {
            "blue": {
                "unit_production": 0,
                "unit_kills": 0,
                "unit_losses": 0,
                "damage_dealt": 0,
                "damage_taken": 0,
                "resources_spent": 0,
                "resources_harvested": 0,
            },
            "red": {
                "unit_production": 0,
                "unit_kills": 0,
                "unit_losses": 0,
                "damage_dealt": 0,
                "damage_taken": 0,
                "resources_spent": 0,
                "resources_harvested": 0,
            },
        }
        self.init_resources = [obs_dict["blue"]["resources"], obs_dict["red"]["resources"]]

    def update(self, new_obs: dict, old_obs: dict):
        for owner in ["blue", "red"]:
            self._update_metric(new_obs, old_obs, owner)
        self.metric["blue"]["damage_dealt"] = self.metric["red"]["damage_taken"]
        self.metric["red"]["damage_dealt"] = self.metric["blue"]["damage_taken"]
        self.metric["blue"]["unit_kills"] = self.metric["red"]["unit_losses"]
        self.metric["red"]["unit_kills"] = self.metric["blue"]["unit_losses"]

    def _update_metric(self, new_obs: dict, old_obs: dict, owner: str):
        for unit_id, unit in new_obs[owner].items():
            if unit_id == "resources":
                continue
            if unit_id not in old_obs[owner]:  # unit create
                self.metric[owner]["unit_production"] += 1
                self.metric[owner]["resources_spent"] += RESOURCES_COST_MAP[unit["type"]]
            else:  # unit alive
                self.metric[owner]["damage_taken"] += old_obs[owner][unit_id]["hp"] - unit["hp"]
        loss_units = [unit_id for unit_id in old_obs[owner].keys() if unit_id not in new_obs[owner].keys()]
        for unit_id in loss_units:
            self.metric[owner]["damage_taken"] += old_obs[owner][unit_id]["hp"]
        self.metric[owner]["unit_losses"] += len(loss_units)
        self.metric[owner]["resources_harvested"] += new_obs[owner]["resources"] - old_obs[owner]["resources"]

    def _update_resources(self, obs: dict):
        self.metric["blue"]["resources_harvested"] = obs["blue"]["resources"] + self.metric["blue"]["resources_spent"] - self.init_resources[0]
        self.metric["red"]["resources_harvested"] = obs["red"]["resources"] + self.metric["red"]["resources_spent"] - self.init_resources[1]

    def display(self, obs: dict):
        self._update_resources(obs)
        output = (f"{'Team':<12} {'Unit Production':<16} {'Unit Kills':<12} {'Unit Losses':<12} "
                f"{'Damage Dealt':<14} {'Damage Taken':<14} {'Resources Spent':<16} {'Resources Harvested':<18}\n")
        output += "-" * 115 + "\n"
        for team, stats in self.metric.items():
            output += (f"{team.capitalize():<12} {stats['unit_production']:<16} {stats['unit_kills']:<12} {stats['unit_losses']:<12} "
                    f"{stats['damage_dealt']:<14} {stats['damage_taken']:<14} {stats['resources_spent']:<16} {stats['resources_harvested']:<18}\n")
        print(output)

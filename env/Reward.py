class Reward:
    def __init__(self, info) -> None:
        """
        [
        "border",
        "obstacle",
        "shelf",
        "cell",
        "change",
        "stay",
        "closer",
        "further",
        "finish",
        ]
        """
        self.border = -1
        self.obstacle = -1
        self.shelf = -1
        self.cell = -1
        self.change = -1
        self.stay = -0.1
        self.closer = 0.2
        self.further = -0.2
        self.finish = 1

    def get_reward(self, info_type):
        return getattr(self, info_type)

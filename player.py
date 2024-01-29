
class Player():
    def __init__(self,squad,position) -> None:
        self.squad = squad
        self.image_position = position
        self.transformed_position = None
        self.soccer_field_position = None
        
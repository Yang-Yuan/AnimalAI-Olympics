class Chaser(object):

    def __init__(self, agent):
        self.agent = agent
        self.newest_is_color = None

    def chase(self):
        self.newest_is_color = self.agent.is_color
        self.chase_internal(self.newest_is_color)

    def chase_in_dark(self):
        imaginary_is_color = self.newest_is_color # TODO enhance
        self.chase_internal(imaginary_is_color)

    def chase_internal(self):
        pass

from utils.logger import LM

class BaseDS:
    def __init__(self, opt):
        self.name = opt['name']
        self.logger = LM('root')

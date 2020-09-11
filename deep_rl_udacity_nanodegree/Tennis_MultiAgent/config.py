class Config:
    def __new__(self):
        """Define this class as a singleton"""
        if not hasattr(self, 'instance'):
            self.instance = super().__new__(self)

            self.instance.device = 'cuda'
            self.instance.seed = 42
            
            self.instance.optimizer_actor = {
                'type' : 'Adam',
                'betas' : [0.9, 0.999],
                'params' : {
                    'lr' : 1e-4,
                    'eps' : 1e-7,
                    'weight_decay' : 0
                }
            }
            
            self.instance.optimizer_critic = {
                'type' : 'Adam',
                'betas' : [0.9, 0.999],
                'params':{
                    'lr' : 1e-3,
                    'eps' : 1e-7,
                    'weight_decay' : 0
                }
            }
            
            self.instance.num_agents = 2
            self.instance.gamma = 0.99
            self.instance.tau = 0.001
            self.instance.buffer_size = 10e6

            self.instance.num_episodes = 15000
            self.instance.batch_size = 128

            self.instance.architecture = [128, 64]
            self.instance.batch_normalization = False

        return self.instance
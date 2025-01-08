class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 28
        self.num_nodes = 28

        self.window_size = 2
        self.time_denpen_len = 25

        self.convo_time_length = 16

        self.kernel_size = 4
        self.stride = 1

        self.hidden_channels = 64
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.1

        # training configs
        self.num_epoch = 100

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 5e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 256
        self.batch_size_test = 16

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True
class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 5

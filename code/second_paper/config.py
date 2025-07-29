class Configs:
    def __init__(self):
        # 定义所有的配置参数及其默认值
        self.task_name = "classification"
        self.seq_len = 20
        self.pred_len = 0
        self.e_layers = 2
        self.enc_in = 132
        self.d_model = 32
        self.embed = "fixed"
        self.freq = "hour"
        self.dropout = 0.1
        self.num_class = 3
        self.top_k=1
        self.d_ff=64
        self.num_kernels=6

    def display(self):
        # 打印所有参数
        print("Configurations:")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

# # 示例使用
# if __name__ == "__main__":
#     configs = Configs()
#     configs.display()

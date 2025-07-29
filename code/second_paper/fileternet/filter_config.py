class Configs:
    def __init__(self):
        # 定义所有的配置参数及其默认值

        self.seq_len = 10
        self.pred_len=3 #分类类别数
        self.enc_in=132
        self.hidden_size=256

    def display(self):
        # 打印所有参数
        print("Configurations:")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

# # 示例使用
# if __name__ == "__main__":
#     configs = Configs()
#     configs.display()

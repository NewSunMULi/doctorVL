import modelscope as ms

if __name__ == '__main__':
    num = int(input("Enter the number of classes:\n1. QWen3\n2. SAM 3"))
    if num == 1:
        model_dir = ms.snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir="model/Qwen/Qwen3-VL-2B-Instruct")
        print(model_dir)

    elif num == 2:
        model_dir = ms.snapshot_download('facebook/sam3', local_dir="./model/sam3/sam3-8b5")
        print(model_dir)

    else:
        raise Exception("Invalid number")
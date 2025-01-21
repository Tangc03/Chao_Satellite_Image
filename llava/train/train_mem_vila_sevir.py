from llava.train.train_vila_sevir import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

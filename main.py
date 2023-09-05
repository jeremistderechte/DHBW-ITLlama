import finetune, preprocess
import pandas as pd
def start_menu():
    github_link = "https://github.com/jeremistderechte/DHBW-ITLlama"
    print(f"\nWelcome to our little tool to finetune meta's llama2 model to function as an IT chatbot for DHBW. "
          f"Visit {github_link} for further documentation.")

    print("\nAttention: \n")

    print("Note that an Nvidia GPU with at least 12GB VRAM is required. On Windows systems it is possible that you need"
          "at least 16GB VRAM, because of the high standard allocation of Windows!\n")

    want_to_start = input("Do you want to continue? (j/n):  ")




    if (want_to_start.upper() == "J"):
        dataset = pd.read_csv("./dataset.csv")
        dataset = preprocess.TunedDataset(dataset)
        dataset = dataset.format_dataset(True)

        model = finetune.Tunedmodel(dataset)
        model.finetune()
    else:
        exit()


if __name__ == '__main__':
    start_menu()


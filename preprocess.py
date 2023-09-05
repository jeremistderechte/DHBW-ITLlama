import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import pandas as pd



class TunedDataset:
    def __init__(self, dataframe_raw):
        self.df = dataframe_raw
        self.formated_dataset = None

    def format_dataset(self, use_context=False):
        self.df.fillna("")

        instruction_list = self.df["instruction"]
        # context_list = df["context"]
        context = "The following is something about the DHBW"
        response_list = self.df["response"]

        text_list = []

        if use_context:
            for instruction, response in zip(instruction_list, response_list):
                text = "<s>[INST] <<SYS>> " + context + " <</SYS>> " + instruction + " [/INST] " + response + " </s>"
                text_list.append(text)
        else:
            for instruction, response in zip(instruction_list, response_list):
                text = "<s>[INST] " + instruction + " [/INST] " + response + " </s>"
                text_list.append(text)

        df_text = pd.DataFrame(columns=["text"])
        df_text["text"] = text_list

        dataset = ds.dataset(pa.Table.from_pandas(df_text).to_batches())

        self.formated_dataset = Dataset(pa.Table.from_pandas(df_text))

        return self.formated_dataset
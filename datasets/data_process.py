import pandas as pd
import json
import sys
import os
from tqdm import tqdm

def main():
    files = os.listdir("/code/zengyufei/datasets/use_for_pretrain")
    new_data = pd.DataFrame(columns=["text"])
    for file in files:
        if not file.endswith(".json"):
            continue
        file_name = "/code/zengyufei/datasets/use_for_pretrain/" + file
        with open(file_name, "r", encoding="utf-8") as f:
            datas = json.loads(f.read())
            
        for idx, d in tqdm(enumerate(datas)):
            new_data.loc[idx, "text"] = d["content"]
            
        break
        
    save_name = "/Users/a58/Downloads/my_test/data/pre_train/all_results.csv"
    new_data.to_csv(save_name, index=False)
    # with open(save_name, "w", encoding="utf-8") as f:
    #     f.write(json.dumps(new_data, ensure_ascii=False, indent=4))



if __name__ == "__main__":
    main()
import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm

data_abs_dir = Path(__file__).parent / "data"


def ttrl_transform_main(args):
    test_filename = args.test_filename
    saved_dir = args.saved_dir
    problem_file = os.path.join(data_abs_dir, test_filename)
    
    test_datas = []
    train_datas = []
    
    with open(problem_file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
        
    for idx, item in enumerate(data):
        assert "prompt_sft" in item
        prompt = item["prompt_sft"]
        answer = "There is no canonical solution for LeeCode benchmark. This is just a placeholder."
        source = 'Leecode-' + test_filename.split(".")[0]
        idx = item["task_id"]
        test = item["test"]
        idx_test = f"test/{source}/{idx}"
        idx_train = f"train/{source}/{idx}"
        test_datas.append({"prompt": prompt, "answer": answer, "source":source, "id": idx_test, "test": test})
        train_datas.append({"prompt": prompt, "answer": answer, "source":source, "id": idx_train, "test": test})
        
        os.makedirs(os.path.join(saved_dir, source), exist_ok=True)
        
        with open(os.path.join(saved_dir, source, 'test.json'), "w", encoding="utf-8") as file:
            json.dump(test_datas, file, indent=4, ensure_ascii=False)
            
        with open(os.path.join(saved_dir, source, 'train.json'), "w", encoding="utf-8") as file:
            json.dump(train_datas, file, indent=4, ensure_ascii=False)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_dir", type=str, default="./ttrl_data/humaneval/", help="output path of your generation")
    parser.add_argument("--test_filename", type=str, default="20240121-Jul.jsonl", help="test filename")
    
    args = parser.parse_args()
    ttrl_transform_main(args)
    pass
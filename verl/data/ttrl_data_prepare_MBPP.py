import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm

data_abs_dir = Path(__file__).parent / "data"

def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
'''.strip().format('\n\n'.join(examples_str), prompt)
        yield {
            'task_id': ex['task_id'],
            'code': ex['code'],
            'test': "\n".join(ex['test_list']),
            'prompt': prompt_with_shots
        }


def ttrl_transform_main(args):
    test_filename = args.test_filename
    saved_dir = args.saved_dir
    problem_file = os.path.join(data_abs_dir, test_filename)
    
    test_datas = []
    train_datas = []
    
    # with open(problem_file, "r", encoding="utf-8") as file:
    #     data = [json.loads(line) for line in file]
    examples = list(read_test_examples(problem_file))
    print("Read {} examples for evaluation over.".format(len(examples)))
        
    for idx, item in enumerate(examples):
        assert "prompt" in item
        prompt =item['prompt']
        answer = item["code"]
        source = test_filename.split(".")[0]
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
    parser.add_argument("--test_filename", type=str, default="MBPP.jsonl", help="test filename")
    
    args = parser.parse_args()
    ttrl_transform_main(args)
    pass
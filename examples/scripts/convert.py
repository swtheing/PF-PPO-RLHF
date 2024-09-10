import sys
import jsonlines


def main(data_path: str):
    """
    python scripts/transfor_sft_data.py {EB_style_data_path:-xxx.jsonl}
        -> output_file: data/xxx.jsonl
    """
    with jsonlines.open(data_path, 'r') as f:
        datas = [data for data in f]

    file_name = data_path.split('/')[-1]
    with jsonlines.open(f'data/{file_name}', 'w') as f:
        for data in datas:
            if 'instruction' in data:
                f.write(data)
                continue
            if 'src' in data:
                if type(data['src']) == str:
                    output = {
                        "instruction": f"{data['src'].strip()}",
                        "input": "",
                        "output": f"{data['tgt'].strip()}" if data['tgt'] else ""
                    }
                else:
                    output = {
                        "instruction": f"{data['src'][0].strip()}",
                        "input": "",
                        "output": f"{data['tgt'][0].strip()}" if data['tgt'] else ""
                    }
                
            else:
                output = { 
                    "instruction": f"{data['query'].strip()}",
                    "input": "",
                    "output": "" if 'output' not in data else f"{data['output'].strip()}"
                    }
            f.write(output)

    print(f'Saved to: data/{file_name}')
    return


if __name__ == '__main__':
    data_path = sys.argv[1]
    main(data_path)

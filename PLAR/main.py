import json
import argparse


def load_args():
    parser = argparse.ArgumentParser()

    # load config file to add default arguments
    with open('configs.json', 'r') as f:
        config = json.load(f)

    parser.add_argument('--engine', type=str, default=config['llm_engine'])
    parser.add_argument('--temperature', type=str, default=config['llm_engine_temperature'])
    parser.add_argument('--max_tokens', type=str, default=config['llm_engine_max_tokens'])

    args = parser.parse_args()

    return args

def main():
    args = load_args()

    

    pass

if __name__ == "__main__":
    main()
from tqdm import tqdm

from pathlib import Path
import os

os.chdir(r"D:\Projects\image_to_text")


def process_files(ip_file_path, op_file_base_path):
    with open(ip_file_path, "r") as fh:
        data = fh.read()
    data = data.split("\n")[23:]
    file = ""
    text = ""
    for line in tqdm(data):
        line = line.split(" ")
        curr_file = "-".join(line[0].split("-")[:-1]) + ".txt"
        curr_text = " ".join(" ".join(line[8:]).split("|")) + " "
        # if len(curr_text) != 0 and curr_text[-1].isalnum():
        #     curr_text = curr_text + " "
        if curr_file != file:
            if file != "":
                with open(op_file_base_path / file, "w") as fh:
                    fh.write(text)
            file = curr_file
            text = curr_text
        else:
            text = text + curr_text


if __name__ == "__main__":
    base_ip = Path("data/IAM/ascii/lines.txt")
    base_op = Path("data/final/text")
    process_files(base_ip, base_op)

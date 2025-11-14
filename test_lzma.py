import os
import lzma
import shutil

for i in os.listdir("model"):
    input_file = f"model/{i}"
    compressed_file = f"compressed_model/{i}.xz"
    # Open the input file and compress it using LZMA
    with open(input_file, "rb") as f_in:
        with lzma.open(compressed_file, "wb", preset=9) as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Compressed {input_file} → {compressed_file}")

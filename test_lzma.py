import lzma
import shutil

# Input and output file names
input_file = "model/lm_head.weight.pt"
compressed_file = "lm_head.weight.pt.xz"

# Open the input file and compress it using LZMA
with open(input_file, "rb") as f_in:
    with lzma.open(compressed_file, "wb", preset=9) as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Compressed {input_file} → {compressed_file}")

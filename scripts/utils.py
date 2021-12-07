import os
import re


def enumerate_over_pairs(x, y):
    for n, i in enumerate(x):
        for m, j in enumerate(y):
            if n == m:
                continue
            yield n, m, i, j


def remove_indices(x, inds):
    """Remove multiple indices from the list in place."""
    inds = sorted(inds)
    for n, i in enumerate(inds):
        x.pop(i - n)


def split_file(filepath, chunk_size=10000000):
    file_number = 0
    with open(filepath, "rb") as f:
        chunk = f.read(chunk_size)
        while chunk:
            with open(filepath + ".part" + str(file_number), "wb") as chunk_file:
                chunk_file.write(chunk)
            file_number += 1
            chunk = f.read(chunk_size)


def concat_file_parts(filepath):
    # collect files
    dir, filename = os.path.split(filepath)
    FILENAME_RE = re.compile(f"{filename}.part([0-9]+)")
    files = []
    for i in os.listdir(dir):
        match = FILENAME_RE.match(i)
        if match:
            number = int(match.groups()[0])
            files.append((i, number))
    # check if all files are present
    files.sort(key=lambda x: x[1])
    for n, i in enumerate(files):
        if n != i[1]:
            raise ValueError(
                f"File {i[0]} has number {i[1]} but has index {n} in the collected files.")
    with open(filepath, "wb") as outfile:
        for (chunkname, _) in files:
            print(f"Appending {chunkname}")
            with open(os.path.join(dir, chunkname), "rb") as chunkfile:
                outfile.write(chunkfile.read())

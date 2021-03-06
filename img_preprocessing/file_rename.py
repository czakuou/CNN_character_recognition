import os
import sys

# change photos names to numbers
def rename(path):
    files = os.listdir(path)
    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))
        print("%s: has been renamed" % file)
        
# run
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("usage: rename.py <path>")
        sys.exit(-1)
        
    rename(sys.argv[1])


if __name__ == "__main__":
    main()
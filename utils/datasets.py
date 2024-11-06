"""Repartitioning the dataset"""

from glob import glob
import os



def partition_dataset(source_dir, dst, ratio=[0.6, 0.2, 0.2]):
    """
    source_dir: str, src dir
    tag: str, target dir
    """
    # get all the npz files

    filelist = glob(os.path.join(source_dir, "*.npz"))

    # generate the trainlist, vallist, testlist

    import random

    random.shuffle(filelist)

    trainlist = filelist[:int(len(filelist)*ratio[0])]

    vallist = filelist[int(len(filelist)*ratio[0]):int(len(filelist)*(ratio[0]+ratio[1]))]

    testlist = filelist[int(len(filelist)*(ratio[0]+ratio[1])):]

    # dump the trainlist, vallist, testlist to txt files

    with open(os.path.join(dst, "trainlist.txt"), "w") as f:
        for item in trainlist:
            f.write("%s\n" % item)

    with open(os.path.join(dst, "vallist.txt"), "w") as f:
        for item in vallist:
            f.write("%s\n" % item)

    with open(os.path.join(dst, "testlist.txt"), "w") as f:
        for item in testlist:
            f.write("%s\n" % item)

    
if __name__ == "__main__":
    source_dir = None

    tag = None
    partition_dataset(source_dir, tag, ratio=[0.8, 0.1, 0.1])
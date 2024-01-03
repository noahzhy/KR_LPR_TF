import os
import random

from tqdm import tqdm


def split_txt_file(file_path, val_ratio):
    # Read the file names from the input txt file
    with open(file_path, 'r', encoding='utf-8') as f:
        file_names = f.readlines()

    # Remove the \n
    # file_names = [file_name.strip() for file_name in file_names if file_name.strip()]
    print('file_names:', len(file_names))
    
    # Shuffle the file names randomly
    random.shuffle(file_names)
    
    # Calculate the number of files to be used for validation based on the given ratio
    num_val = int(len(file_names) * val_ratio)
    
    # Split the shuffled file names into two lists: one for train and one for val
    val_files = file_names[:num_val]
    train_files = file_names[num_val:]
    
    # Write the train file names to a new txt file called train.txt
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_files)
    
    # Write the val file names to a new txt file called val.txt
    with open('val.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_files)


def move_via_txt(txt_path, src_dir, dst_dir):
    # Read the file names from the input txt file
    with open(txt_path, 'r', encoding='utf-8') as f:
        file_names = f.readlines()

    print('file_names:', len(file_names))

    for file_name in tqdm(file_names):
        file_name = file_name.strip()
        img_path = os.path.join(src_dir, file_name + '.jpg')
        npy_path = os.path.join(src_dir, file_name + '.npy')
        label_path = os.path.join(src_dir, file_name + '_label.npy')
        move2dir(img_path, dst_dir)
        move2dir(npy_path, dst_dir)
        move2dir(label_path, dst_dir)


# move file to dir
def move2dir(file_path, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # get file name
    file_name = os.path.basename(file_path)
    # get file new path
    new_path = os.path.join(dir_path, file_name)
    # move file, if file exists, overwrite it
    if os.path.exists(new_path):
        os.remove(new_path)
    os.rename(file_path, new_path)



if __name__ == '__main__':
    # split_txt_file('train_val_checked.txt', 0.1)

    # txt = 'train.txt'
    # src_dir = 'E:/dataset/license_plate_chars/v2'
    # dst_dir = 'E:/dataset/license_plate_chars/v2/train'
    # move_via_txt(txt, src_dir, dst_dir)

    txt = 'val.txt'
    src_dir = 'E:/dataset/license_plate_chars/v2'
    dst_dir = 'E:/dataset/license_plate_chars/v2/val'
    move_via_txt(txt, src_dir, dst_dir)
    

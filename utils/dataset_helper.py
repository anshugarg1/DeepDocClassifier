import os
from definitions import train_id_path, val_id_path, test_id_path

def create_label_dict(dataset_path):
    #Create mapping between classs name and class label.
    class_lbl_dict = {}
    for i, folder in enumerate(os.listdir(dataset_path)):
        class_lbl_dict[folder] = i

    # print("Folder and class label mapping dict: ", class_lbl_dict)
    return class_lbl_dict


def gen_datset_split_file(args):
    #Generate dataset splitting among train, val, test and store image ids to the corresponding python files.
    train_ls = []
    val_ls = []
    test_ls = []

    val_data_count = int(args.train_per_cls*args.valSplit)
    train_data_count = args.train_per_cls - val_data_count

    for folder in os.listdir(args.dataset_path):
        img_path = os.path.join(args.dataset_path, folder)
        print("Image folder path: ", img_path)

        for i, image_id in enumerate(os.listdir(img_path)):
            if '.jpg' in image_id:
                if i < train_data_count:
                    train_ls.append(os.path.join(folder,image_id))
                    # train_ls.append(image_id)
                elif i>=train_data_count and i<args.train_per_cls:
                    val_ls.append(os.path.join(folder,image_id))
                    # val_ls.append(image_id)
                else:
                    test_ls.append(os.path.join(folder,image_id))
                    # test_ls.append(image_id)
        
        print("Training list length: ", len(train_ls))
        print("Validation list length: ", len(val_ls))
        print("Test list length: ", len(test_ls))
    
    with open(train_id_path,'w') as f:
        for file_name in train_ls:
            f.write('%s\n'%file_name)

    with open(val_id_path,'w') as f:
        for file_name in val_ls:
            f.write('%s\n'%file_name)

    with open(test_id_path,'w') as f:
        for file_name in test_ls:
            f.write('%s\n'%file_name)


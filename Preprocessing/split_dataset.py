from Preprocessing.preprocessing import *
from parameters import meta
# from configuration import *
# from parameters import MetaParameters
# from Evaluation.evaluation import *
import json
from pprint import pprint


#########################################################################################################################
# Create subject list and after shuffling it, split to train, valid and test sets
#########################################################################################################################
def create_folds_list():
    dataset_list = ReadImages(f'{meta.ORIGS_DIR}').get_dataset_list()

    try:
        dataset_list.remove('.DS_Store')
    except ValueError:
        pass

    random.shuffle(dataset_list)
    random.shuffle(dataset_list)

    dataset_size = len(dataset_list)
    test_list = dataset_list[round(0.8 * dataset_size):]
    train_list  = list(set(dataset_list) - set(test_list))

    train_dataset_size = len(train_list)

    valid_list_01 = train_list[round(0.8*train_dataset_size):]
    train_list_01 = list(set(train_list)-set(valid_list_01))

    valid_list_02 = train_list[round(0.6*train_dataset_size):round(0.8*train_dataset_size)]
    train_list_02 = list(set(train_list)-set(valid_list_02))

    valid_list_03 = train_list[round(0.4*train_dataset_size):round(0.6*train_dataset_size)]
    train_list_03 = list(set(train_list)-set(valid_list_03))

    valid_list_04 = train_list[round(0.2*train_dataset_size):round(0.4*train_dataset_size)]
    train_list_04 = list(set(train_list)-set(valid_list_04))

    valid_list_05 = train_list[:round(0.2*train_dataset_size)]
    train_list_05 = list(set(train_list)-set(valid_list_05))

    folds_list = {
                'train_list_01' : train_list_01,
                'valid_list_01' : valid_list_01,
                'train_list_02' : train_list_02,
                'valid_list_02' : valid_list_02,
                'train_list_03' : train_list_03,
                'valid_list_03': valid_list_03,
                'train_list_04' : train_list_04,
                'valid_list_04': valid_list_04,
                'train_list_05' : train_list_05,
                'valid_list_05' : valid_list_05,
                'train_list_full' : train_list,
                'valid_list_full' : test_list,
                'test_list' : test_list,
                }

    with open(f'{meta.DATASET_DIR}{meta.DATASET_NAME}_folds_list.json', "w") as fdct:
        json.dump(folds_list, fdct) # записываем структуру в файл


if not os.path.exists(f'{meta.DATASET_DIR}{meta.DATASET_NAME}_folds_list.json'):
    create_folds_list()

try:
    with open(f'{meta.DATASET_DIR}{meta.DATASET_NAME}_folds_list.json', "r") as fdct:
        folds_dict = json.load(fdct)

        train_list = folds_dict[f'train_list_{meta.FOLD_NAME}']
        valid_list = folds_dict[f'valid_list_{meta.FOLD_NAME}']
        test_list = folds_dict[f'test_list']

        pprint(f'test_list = {test_list}')
        pprint(f'valid_list = {valid_list}')
        pprint(f'train_list = {train_list}')
except:
    print('LOOK AT JSON FILE IN DATASET DIRECTORY')


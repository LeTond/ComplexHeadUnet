from configuration import *
from Validation.metrics import *

from Training.dataset import *
from Preprocessing.split_dataset import *


class PlotResults(MetaParameters):

    def __init__(self):         
        super(MetaParameters, self).__init__()
        self.kernel_sz = self.KERNEL
        self.dict_class_stats = self.create_dict_class()

    def data_loader(self, data_list, kernel_sz, augmentation=False):
        getds = GetData(data_list, augmentation).generated_data_list()
        getds_origin = getds[0]
        getds_mask = getds[1]
        getds_names = getds[2]
        data_set = MyDataset(self.NUM_CLASS, getds_origin, getds_mask, getds_names, kernel_sz, default_transform)
        
        data_batch_size = len(data_set)
        data_loader = DataLoader(data_set, data_batch_size, drop_last=True, shuffle=False, pin_memory=True)

        return data_loader

    def create_dict_class(self):
        dict_class_stats = {}

        for key in range(1, self.NUM_CLASS): 
            dict_class_stats[f'Precision_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Recall_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Accuracy_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Dice_{self.DICT_CLASS[key]}'] = []

            dict_class_stats[f'FN_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'FP_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'GTPix_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'CMPix_{self.DICT_CLASS[key]}'] = []

            dict_class_stats[f'GTVol_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'CMVol_{self.DICT_CLASS[key]}'] = []

        return dict_class_stats

    def bland_altman_per_subject(self, model, test_list, meta, kernel_sz):
        for subj in test_list:
            try:
                for key in range(1, self.NUM_CLASS): 
                    data_loader = self.data_loader([subj], kernel_sz, False)
                    tm = TissueMetrics(model, data_loader)
                    dict_class_sub_stats = tm.bland_altman_metrics()
                    
                    self.dict_class_stats[f'GTVol_{self.DICT_CLASS[key]}'].append(np.sum(dict_class_sub_stats[f'GTVol_{self.DICT_CLASS[key]}']))
                    self.dict_class_stats[f'CMVol_{self.DICT_CLASS[key]}'].append(np.sum(dict_class_sub_stats[f'CMVol_{self.DICT_CLASS[key]}']))
            
            except ValueError:
                print(f'Subject {subj} has no suitable images !!!!')

        return self.dict_class_stats

    def stats_per_subject(self, model, test_list, meta, kernel_sz):
        for subj in test_list:
            try:
                data_loader = self.data_loader([subj], kernel_sz, False)
                tm = TissueMetrics(model, data_loader)
                dict_class_sub_stats = tm.image_metrics()

                for key in range(1, self.NUM_CLASS):
                    self.dict_class_stats[f'Precision_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Precision_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Recall_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Recall_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Accuracy_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Accuracy_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Dice_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Dice_{self.DICT_CLASS[key]}']

            except ValueError:
                print(f'Subject {subj} has no suitable images !!!!')

        return self.dict_class_stats

    def prepare_plot(self, sub_names, origImage, origMask, predMask, dice_layers):
        figure, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (12, 12))

        origImage = np.resize(origImage.cpu(), (self.kernel_sz, self.kernel_sz))
        predMask = np.resize(predMask.cpu(), (self.kernel_sz, self.kernel_sz))
        origMask = np.resize(origMask.cpu(), (self.kernel_sz, self.kernel_sz))

        for key in range(1, self.NUM_CLASS):
            predMask[0][key-1] = key
            origMask[0][key-1] = key

        ax[0].imshow(origImage, plt.get_cmap('gray'))
        ax[1].imshow(origImage, plt.get_cmap('gray'))
        # ax[1].contour(origMask, alpha = 0.5)
        ax[1].imshow(origMask, alpha = 0.5)
        ax[2].imshow(origImage, plt.get_cmap('gray'))
        ax[2].imshow(predMask, alpha = 0.5)
        ax[3].imshow(predMask, alpha = 0.5)

        ax[0].set_title(f"{sub_names}", fontsize = 10, fontweight = 'bold')
        ax[1].set_title(f"Dice: {dice_layers} \nManual mask", fontsize = 10, fontweight ='bold')
        ax[2].set_title(f"Computed mask", fontsize = 10, fontweight='bold')
        ax[3].set_title(f"Computed mask", fontsize = 10, fontweight='bold')
        
        figure.set_edgecolor("green")
        figure.tight_layout()
        
        return figure

    def show_predicted(self, predicted_masks):
        for i in range(predicted_masks[0][0]):
            dice_layers = str('')

            for key in range(1, self.NUM_CLASS):
                dice_layers += f' {self.DICT_CLASS[key]} = '
                dice_layers += str(round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3))

            self.prepare_plot(predicted_masks[1][i], predicted_masks[2][i], predicted_masks[3][i], predicted_masks[4][i], dice_layers)


model = torch.load(f'{meta.UNET1_PROJ_NAME}/{meta.MODEL_NAME}.pth').to(device=device)
kernel_sz = meta.KERNEL

pltres = PlotResults()
test_loader = pltres.data_loader(test_list, kernel_sz, False)

ds = DiceLoss()
show_predicted_masks = MaskPrediction().prediction_masks(model, test_loader)
tm = TissueMetrics(model, test_loader)

try:
    bland_dict_class_stats = pltres.bland_altman_per_subject(model, test_list, meta, kernel_sz)
    dict_class_stats = pltres.stats_per_subject(model, test_list, meta, kernel_sz)

except ValueError:
    print(f'Subjects has no suitable images !!!!')



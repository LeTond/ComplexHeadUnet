from configuration import *
from parameters import MetaParameters
from Validation.metrics import DiceLoss
from tqdm.notebook import tqdm


ds = DiceLoss()


class TrainNetwork(MetaParameters):

    def __init__(self, model, optimizer, loss_function, train_loader, valid_loader, meta, ds):         
        super(MetaParameters, self).__init__()
        self.ds = ds 
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.project_name = self.UNET1_PROJ_NAME

    def get_metrics(self, loader_):
        self.model.eval()

        loss = 0
        num_batches = len(loader_)
        num_layers = len(self.DICT_CLASS)   
        dictionary = {}

        for key in range(1, num_layers): 
            dictionary[f'Dice_{self.DICT_CLASS[key]}'] = 0

        with torch.no_grad():
            
            for inputs, labels, sub_names in loader_:
                inputs, labels, sub_names = inputs.to(device), labels.to(device), list(sub_names)   

                predict = self.model(inputs)
                loss += self.loss_function(predict, labels)

                predict = torch.softmax(predict, dim=1)
                predict = torch.argmax(predict, dim=1)
                labels = torch.argmax(labels, dim=1)
                
                for key in range(1, num_layers):
                    predict_ = (predict == key)
                    labels_ = (labels == key)
                    dictionary[f'Dice_{self.DICT_CLASS[key]}'] += float(self.ds(predict_, labels_))

        for key in range(1, num_layers):
            dictionary[f'Dice_{self.DICT_CLASS[key]}'] /= num_batches
        mean_loss = float((loss / num_batches))

        return mean_loss, dictionary

    def train(self):
        trigger_times, the_last_loss = 0, 100

        for epoch in range(self.EPOCHS + 1):
            results = ''
            time_start_epoch = time.time()
            self.model.train()
            
            for inputs, labels, sub_names in self.train_loader:
                inputs, labels, sub_names = inputs.to(device), labels.to(device), list(sub_names)   
          
                predict = self.model(inputs)
                train_loss = loss_function(predict, labels)

                predict = torch.softmax(predict, dim=1)
                predict = torch.argmax(predict, dim=1)
                labels = torch.argmax(labels, dim=1)
                
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                
            scheduler_gen.step() #g_mean_train_loss,  g_mean_valid_loss
            
            training = self.get_metrics(self.train_loader)
            validating = self.get_metrics(self.valid_loader)
            
            num_layers = len(self.DICT_CLASS)
            results += f'TRAIN: Loss = {round(training[0], 3)}'
            for key in range(1, num_layers):
                results += f' Dice_{self.DICT_CLASS[key]} = ' + str(round(training[1].get(f'Dice_{self.DICT_CLASS[key]}'), 3))
            results += f'\nVALID: Loss = {round(validating[0], 3)}'
            for key in range(1, num_layers):
                results += f' Dice_{self.DICT_CLASS[key]} = ' + str(round(validating[1].get(f'Dice_{self.DICT_CLASS[key]}'), 3))
            
            log_stats(results, self.project_name)

            if validating[0] > the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)

                if trigger_times >= self.EARLY_STOPPING:
                    print('Early stopping!\nStart to test process.')
                    return self.model

            else:
                trigger_times = 0

            if validating[0] <= the_last_loss:
                the_last_loss = validating[0]
                torch.save(self.model, f'{self.project_name}/model_best.pth')
                print(f'{self.project_name}/model_best - {epoch} saved!')

            print(results)
            time_end_epoch = time.time()
            print(f'Epoch time: {round(time_end_epoch - time_start_epoch)} seconds') 
            
        return self.model



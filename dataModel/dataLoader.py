from typing import Callable, Dict

from dataModel.datamd_ import DpDataSet
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DpDataMoDule(LightningDataModule):
    def __init__(self,config: Dict) -> None:
        super(DpDataMoDule, self).__init__()
        self.train_data_path = config['data_train_path'] 
        self.val_data_path = config['data_test_path']
        
        # self.batch_size = config.get('batch_size', 1) 
        self.save_hyperparameters()
        self.prepare_data()

    def setup(self, stage: str = None) -> None:
        pass

    def prepare_data(self) -> None:
        # trainData = DpDataSet(path=self.data_path + '/data_train.jsonl')
        # valData = DpDataSet(path=self.data_path + '/data_val.jsonl')
        trainData = DpDataSet(self.train_data_path)
        testData = DpDataSet(self.val_data_path)
        self.train_loader = DataLoader(trainData, batch_size= 1, shuffle= True, num_workers= 0)
        self.val_loader = DataLoader(testData, batch_size= 1, shuffle= False, num_workers= 0)
        # return super().prepare_data()
    
    def train_dataloader(self) -> None:
        return self.train_loader
    
    def val_dataloader(self) -> None:
        return self.val_loader
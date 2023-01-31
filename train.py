# To be continue ...
from typing import Dict
import torch
import mlflow
from dataModel.dataLoader import DpDataMoDule
from dotenv import load_dotenv
from helpers import load_config
from modeling.base_model import LitBaseParsing
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

from datetime import datetime
load_dotenv()

config: Dict = load_config("main.yaml")
logger = MLFlowLogger('eKyC/DP', tracking_uri="http://10.10.1.37:5000")
mlflow.set_tracking_uri("http://10.10.1.37:5000")

data_module = DpDataMoDule(config=config)
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(accelerator='gpu',
                  devices=1,
                  logger=logger,
                  max_epochs=81,
                  auto_lr_find=False,
                  auto_scale_batch_size='binsearch',
                  callbacks=[lr_monitor],
                  log_every_n_steps=5
                  )


DpModel = LitBaseParsing()
trainer.fit(model=DpModel, datamodule=data_module)

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")
print('Export last .pt')
with open(f"./resources/checkpoints/DP_model_finetune_{now}.pt", "wb") as f:
    torch.save(DpModel.state_dict(), f)

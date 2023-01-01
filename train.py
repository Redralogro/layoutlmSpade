# To be continue ...
from datetime import datetime
from typing import Dict

import mlflow
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from transformers import AutoTokenizer
from helpers import load_config
from dataModel.dataLoader import DpDataMoDule
from model import spadeLayoutLM
from modeling.warped_model import LitLayoutParsing

config: Dict = load_config("main.yaml")
logger = MLFlowLogger("eKyC/DP", tracking_uri="http://10.10.1.37:5000")
mlflow.set_tracking_uri("http://10.10.1.37:5000")


data_module = DpDataMoDule(config=config)
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(accelerator='gpu',
                  devices=1,
                  logger=logger,
                  max_epochs=200,
                  auto_lr_find=True,
                  auto_scale_batch_size='binsearch',
                  callbacks=[lr_monitor],
                  log_every_n_steps=5
                  )

# DpModel = spadeLayoutLM()
DpModel = LitLayoutParsing()
trainer.fit(model=DpModel, datamodule=data_module)
now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")
with open(f"./resources/checkpoints/DP_{now}.pt", "wb") as f:
    torch.save(DpModel.state_dict(), f)

# To be continue ...
from datetime import datetime
from typing import Dict
import json
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
from os import environ as env

from dotenv import load_dotenv
load_dotenv()

config: Dict = load_config("main.yaml")
logger = MLFlowLogger(env['EXP'], tracking_uri=env['HOSTNAME'])
mlflow.set_tracking_uri(env['HOSTNAME'])

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
data_module = DpDataMoDule(config=config)
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(accelerator='gpu',
                  devices=1,
                  logger=logger,
                  max_epochs=301,
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
# print('Export .pt')
# with open(f"./resources/checkpoints/DP_{now}.pt", "wb") as f:
#     torch.save(DpModel.state_dict(), f)
# print("Model's state_dict:")
# for param_tensor in DpModel.state_dict():
#     print(param_tensor, "\t",DpModel.state_dict()[param_tensor])




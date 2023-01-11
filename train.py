# To be continue ...
import json
from datetime import datetime
from os import environ as env
from typing import Dict

import mlflow
import torch
from dataModel.dataLoader import DpDataMoDule
from dotenv import load_dotenv
from helpers import load_config
from model import spadeLayoutLM
from modeling.base_model import LitBaseParsing
from modeling.warped_model import LitLayoutParsing
from modeling.layout_model import LitModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from transformers import AutoTokenizer

load_dotenv()

config: Dict = load_config("main.yaml")
logger = MLFlowLogger('eKyC/DP', tracking_uri="http://10.10.1.37:5000")
mlflow.set_tracking_uri("http://10.10.1.37:5000")

tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
data_module = DpDataMoDule(config=config)
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(accelerator='gpu',
                  devices=1,
                  logger=logger,
                  #   resume_from_checkpoint='./resources/checkpoints/DP_model_finetune_05-01-2023_14-07-10.pt',
                  max_epochs=301,
                  auto_lr_find=True,
                  auto_scale_batch_size='binsearch',
                  callbacks=[lr_monitor],
                  log_every_n_steps=5
                  )

# DpModel = spadeLayoutLM()
# DpModel = LitLayoutParsing()
# model = LitLayoutParsing()
# model.load_state_dict(torch.load('./resources/checkpoints/DP_model_03-01-2023_10-47-23.pt'))
DpModel = LitBaseParsing()
# DpModel = LitModel(config['model'])
trainer.fit(model=DpModel, datamodule=data_module)

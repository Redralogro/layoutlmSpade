# To be continue ...
from typing import Dict

import mlflow
from dataModel.dataLoader import DpDataMoDule
from dotenv import load_dotenv
from helpers import load_config
from modeling.base_model import LitBaseParsing
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

load_dotenv()

config: Dict = load_config("main.yaml")
logger = MLFlowLogger('eKyC/DP', tracking_uri="http://10.10.1.37:5000")
mlflow.set_tracking_uri("http://10.10.1.37:5000")

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


DpModel = LitBaseParsing()
trainer.fit(model=DpModel, datamodule=data_module)

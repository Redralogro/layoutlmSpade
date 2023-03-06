from typing import Dict
import torch
import mlflow
from dataModel.dataLoader import DpDataMoDule
from dotenv import load_dotenv
from helpers import load_config
from models.layout_parsing import LitBaseParsing
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

from datetime import datetime
load_dotenv()


config: Dict = load_config("main.yaml")
onnx_config = config['onnx_exporter']
logger = MLFlowLogger('eKyC/DP', tracking_uri="http://10.10.1.37:5000")
mlflow.set_tracking_uri("http://10.10.1.37:5000")



<<<<<<< HEAD
# data_module = DpDataMoDule(config=config)
=======
data_module = DpDataMoDule(config=config)
>>>>>>> 9f1dd023b8eacd9eb6316eed4264a880142005c9
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = Trainer(accelerator='gpu',
                  devices=1,
                  logger=logger,
                  max_epochs=101,
                  auto_lr_find=False,
                  auto_scale_batch_size='binsearch',
                  callbacks=[lr_monitor],
                  log_every_n_steps=5
                  )

DpModel = LitBaseParsing()
<<<<<<< HEAD
# trainer.fit(model=DpModel, datamodule=data_module)

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")
# print('Export last .pt')
# with open(f"./resources/checkpoints/DP_model_finetune_{now}.pt", "wb") as f:
#     torch.save(DpModel.state_dict(), f)


dummy_input = torch.zeros(3,126).cuda()
=======
trainer.fit(model=DpModel, datamodule=data_module)

now = datetime.now()
now = now.strftime("%d-%m-%Y_%H-%M-%S")
print('Export last .pt')
with open(f"./resources/checkpoints/DP_model_finetune_{now}.pt", "wb") as f:
    torch.save(DpModel.state_dict(), f)


dummy_input = torch.zeros(1,126).cuda()
>>>>>>> 9f1dd023b8eacd9eb6316eed4264a880142005c9
dummy_boxes = torch.zeros(1,126,4).cuda()
dummy_maps = torch.zeros(1,126).cuda()


symbolic_names = {0: "batch_size", 1: "max_seq_len"}
# try :
torch.onnx.export(DpModel.cuda(),
<<<<<<< HEAD
                    args =(dummy_input, dummy_boxes, dummy_maps),
=======
                    args =(dummy_input,dummy_input,dummy_input, dummy_boxes, dummy_maps),
>>>>>>> 9f1dd023b8eacd9eb6316eed4264a880142005c9
                    f = f"{onnx_config['path']}LitLP_{now}_fp32.onnx",
                    export_params=True,
                    opset_version=12,
                    verbose=True,
<<<<<<< HEAD
                    input_names=["input_", "bbox", "maps"],
                    output_names=["output"],
                    dynamic_axes={
                    'input_': symbolic_names,
=======
                    input_names=["input_ids","attention_mask", "token_type_ids", "bbox", "maps"],
                    output_names=["output"],
                    dynamic_axes={
                    'input_ids': symbolic_names,
                    'input_mask' : symbolic_names,
                    'segment_ids' : symbolic_names,
>>>>>>> 9f1dd023b8eacd9eb6316eed4264a880142005c9
                    "bbox": symbolic_names,
                    "maps": symbolic_names,
                    }
                    )
print(" Export onnx model successfully !!!")
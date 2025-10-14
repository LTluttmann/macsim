import hydra
import pyrootutils
from dataclasses import asdict
import lightning.pytorch as pl
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from rl4co.utils import pylogger

from macsim.algorithms.base import EvalModule, ActiveSearchModule
from macsim.utils.utils import hydra_run_wrapper, get_wandb_logger
from macsim.utils.config import (
    EnvParams,
    EnvParamList,
    TrainingParams, 
    TestParams
)


pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)
# A logger for this file
log = pylogger.get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../configs/", config_name="main")
@hydra_run_wrapper
def main(cfg: DictConfig):
    
    env_list = getattr(cfg.env, "list", None)
    if env_list is not None:
        instance_params = EnvParamList.initialize(cfg.env)
    else:
        instance_params = EnvParams.initialize(**cfg.env)
    trainer_params = TrainingParams(**cfg.train)
    test_params = TestParams(**cfg.test)

    hc = HydraConfig.get()
    pl.seed_everything(test_params.seed)


    if test_params.active_search_params is not None:
        PlModule = ActiveSearchModule
    else:
        PlModule = EvalModule

    model, model_params = PlModule.init_from_checkpoint(
        test_params.checkpoint, 
        instance_params,
        test_params=test_params,
    )
    
    if cfg.get("logger", None) is not None:
        log.info("Instantiating loggers...")
        logger = get_wandb_logger(cfg, model_params, hc, training=False)
    else:
        logger = None

    trainer = Trainer(
        accelerator=trainer_params.accelerator,
        devices=trainer_params.devices,
        logger=logger,
        default_root_dir=hc.runtime.output_dir if hc else None,
    )


    for logger in trainer.loggers:
        hparams = {
            "model": asdict(model_params),
            "test": asdict(test_params)
        }
        logger.log_hyperparams(hparams)

    trainer.test(model)


if __name__ == "__main__":
     main()
from models.human_detection import HumanModel
from models.firearm_detection import FirearmModel
from core.association import Grid
from core.pipeline import Pipeline
from core.logger import CSVLogger
from core.evidence import EvidenceSaver
from core.api_publisher import ApiPublisher

class PipelineService:
    def __init__(self, config):
       self.pipeline = Pipeline(
        HumanModel(),
        FirearmModel(),
        CSVLogger(),
        Grid(grid_size=config["grid"]["size"]),
        config["thresholds"],
        EvidenceSaver(),
        ApiPublisher(config)
    )



    def run(self, frame, camera_id="API"):
        return self.pipeline.process(frame, camera_id)

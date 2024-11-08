import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent)) # hack to get around grpc_python's regid design

from biopb.lacss.lacss_pb2_grpc import Lacss, LacssServicer, LacssStub, add_LacssServicer_to_server
from biopb.lacss.detection_request_pb2 import DetectionRequest
from biopb.lacss.detection_response_pb2 import DetectionResponse, ScoredROI
from biopb.lacss.bindata_pb2 import BinData
from biopb.lacss.detection_settings_pb2 import DetectionSettings
from biopb.lacss.image_data_pb2 import ImageData, Pixels, ImageAnnotation
from biopb.lacss.roi_pb2 import ROI, Rectangle, Mask, Mesh, Polygon, Point

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: lacss/detection_settings.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'lacss/detection_settings.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1elacss/detection_settings.proto\x12\x0b\x62iopb.lacss\"\x8e\x02\n\x11\x44\x65tectionSettings\x12\x1a\n\rmin_cell_area\x18\x01 \x01(\x02H\x01\x88\x01\x01\x12\x16\n\tmin_score\x18\x02 \x01(\x02H\x02\x88\x01\x01\x12#\n\x16segmentation_threshold\x18\x03 \x01(\x02H\x03\x88\x01\x01\x12\x14\n\x07nms_iou\x18\x04 \x01(\x02H\x04\x88\x01\x01\x12\x1c\n\x12\x63\x65ll_diameter_hint\x18\x06 \x01(\x02H\x00\x12\x16\n\x0cscaling_hint\x18\x07 \x01(\x02H\x00\x42\r\n\x0bresize_infoB\x10\n\x0e_min_cell_areaB\x0c\n\n_min_scoreB\x19\n\x17_segmentation_thresholdB\n\n\x08_nms_iouB\x0f\n\x0b\x62iopb.lacssP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'lacss.detection_settings_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\013biopb.lacssP\001'
  _globals['_DETECTIONSETTINGS']._serialized_start=48
  _globals['_DETECTIONSETTINGS']._serialized_end=318
# @@protoc_insertion_point(module_scope)

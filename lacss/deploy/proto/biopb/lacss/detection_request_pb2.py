# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: biopb/lacss/detection_request.proto
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
    'biopb/lacss/detection_request.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import field_mask_pb2 as google_dot_protobuf_dot_field__mask__pb2
from biopb.lacss import image_data_pb2 as biopb_dot_lacss_dot_image__data__pb2
from biopb.lacss import detection_settings_pb2 as biopb_dot_lacss_dot_detection__settings__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n#biopb/lacss/detection_request.proto\x12\x0b\x62iopb.lacss\x1a google/protobuf/field_mask.proto\x1a\x1c\x62iopb/lacss/image_data.proto\x1a$biopb/lacss/detection_settings.proto\"\xaa\x01\n\x10\x44\x65tectionRequest\x12*\n\nimage_data\x18\x01 \x01(\x0b\x32\x16.biopb.lacss.ImageData\x12:\n\x12\x64\x65tection_settings\x18\x02 \x01(\x0b\x32\x1e.biopb.lacss.DetectionSettings\x12.\n\nfield_mask\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.FieldMaskB\x0f\n\x0b\x62iopb.lacssP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'biopb.lacss.detection_request_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\013biopb.lacssP\001'
  _globals['_DETECTIONREQUEST']._serialized_start=155
  _globals['_DETECTIONREQUEST']._serialized_end=325
# @@protoc_insertion_point(module_scope)

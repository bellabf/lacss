# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: lacss/roi.proto
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
    'lacss/roi.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import bindata_pb2 as lacss_dot_bindata__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0flacss/roi.proto\x12\x0b\x62iopb.lacss\x1a\x13lacss/bindata.proto\"3\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\x0e\n\x01z\x18\x03 \x01(\x02H\x00\x88\x01\x01\x42\x04\n\x02_z\"[\n\tRectangle\x12$\n\x08top_left\x18\x01 \x01(\x0b\x32\x12.biopb.lacss.Point\x12(\n\x0c\x62ottom_right\x18\x02 \x01(\x0b\x32\x12.biopb.lacss.Point\"Y\n\x04Mask\x12)\n\trectangle\x18\x01 \x01(\x0b\x32\x16.biopb.lacss.Rectangle\x12&\n\x08\x62in_data\x18\x02 \x01(\x0b\x32\x14.biopb.lacss.BinData\"Q\n\x07\x45llipse\x12\"\n\x06\x63\x65nter\x18\x01 \x01(\x0b\x32\x12.biopb.lacss.Point\x12\"\n\x06radius\x18\x02 \x01(\x0b\x32\x12.biopb.lacss.Point\"-\n\x07Polygon\x12\"\n\x06points\x18\x01 \x03(\x0b\x32\x12.biopb.lacss.Point\"\xb1\x01\n\x04Mesh\x12!\n\x05verts\x18\x01 \x03(\x0b\x32\x12.biopb.lacss.Point\x12%\n\x05\x66\x61\x63\x65s\x18\x02 \x03(\x0b\x32\x16.biopb.lacss.Mesh.Face\x12#\n\x07normals\x18\x03 \x03(\x0b\x32\x12.biopb.lacss.Point\x12\x0e\n\x06values\x18\x04 \x03(\x02\x1a*\n\x04\x46\x61\x63\x65\x12\n\n\x02p1\x18\x01 \x01(\r\x12\n\n\x02p2\x18\x02 \x01(\r\x12\n\n\x02p3\x18\x03 \x01(\r\"\xf8\x01\n\x03ROI\x12#\n\x05point\x18\x01 \x01(\x0b\x32\x12.biopb.lacss.PointH\x00\x12+\n\trectangle\x18\x02 \x01(\x0b\x32\x16.biopb.lacss.RectangleH\x00\x12\'\n\x07\x65llipse\x18\x03 \x01(\x0b\x32\x14.biopb.lacss.EllipseH\x00\x12\'\n\x07polygon\x18\x04 \x01(\x0b\x32\x14.biopb.lacss.PolygonH\x00\x12!\n\x04mesh\x18\x05 \x01(\x0b\x32\x11.biopb.lacss.MeshH\x00\x12!\n\x04mask\x18\x06 \x01(\x0b\x32\x11.biopb.lacss.MaskH\x00\x42\x07\n\x05shapeB\x0f\n\x0b\x62iopb.lacssP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'lacss.roi_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\013biopb.lacssP\001'
  _globals['_POINT']._serialized_start=53
  _globals['_POINT']._serialized_end=104
  _globals['_RECTANGLE']._serialized_start=106
  _globals['_RECTANGLE']._serialized_end=197
  _globals['_MASK']._serialized_start=199
  _globals['_MASK']._serialized_end=288
  _globals['_ELLIPSE']._serialized_start=290
  _globals['_ELLIPSE']._serialized_end=371
  _globals['_POLYGON']._serialized_start=373
  _globals['_POLYGON']._serialized_end=418
  _globals['_MESH']._serialized_start=421
  _globals['_MESH']._serialized_end=598
  _globals['_MESH_FACE']._serialized_start=556
  _globals['_MESH_FACE']._serialized_end=598
  _globals['_ROI']._serialized_start=601
  _globals['_ROI']._serialized_end=849
# @@protoc_insertion_point(module_scope)

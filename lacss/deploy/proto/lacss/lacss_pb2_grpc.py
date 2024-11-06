# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from . import detection_request_pb2 as lacss_dot_detection__request__pb2
from . import detection_response_pb2 as lacss_dot_detection__response__pb2

GRPC_GENERATED_VERSION = '1.66.2'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in lacss/lacss_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class LacssStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RunDetection = channel.unary_unary(
                '/biopb.lacss.Lacss/RunDetection',
                request_serializer=lacss_dot_detection__request__pb2.DetectionRequest.SerializeToString,
                response_deserializer=lacss_dot_detection__response__pb2.DetectionResponse.FromString,
                _registered_method=True)
        self.RunDetectionStream = channel.stream_stream(
                '/biopb.lacss.Lacss/RunDetectionStream',
                request_serializer=lacss_dot_detection__request__pb2.DetectionRequest.SerializeToString,
                response_deserializer=lacss_dot_detection__response__pb2.DetectionResponse.FromString,
                _registered_method=True)


class LacssServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RunDetection(self, request, context):
        """Unitary call for computing cell detection / segmentation
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RunDetectionStream(self, request_iterator, context):
        """The streaming version allows dynamic updating of the DetectionRequest and
        getting updated results. For example, a client may request analyses at
        several different settings on the same image. After initial request, the
        following streaming requests no long need to transmit the image pixel
        data anymore, but only the new parameter settings, which saves network
        bandwidth.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LacssServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RunDetection': grpc.unary_unary_rpc_method_handler(
                    servicer.RunDetection,
                    request_deserializer=lacss_dot_detection__request__pb2.DetectionRequest.FromString,
                    response_serializer=lacss_dot_detection__response__pb2.DetectionResponse.SerializeToString,
            ),
            'RunDetectionStream': grpc.stream_stream_rpc_method_handler(
                    servicer.RunDetectionStream,
                    request_deserializer=lacss_dot_detection__request__pb2.DetectionRequest.FromString,
                    response_serializer=lacss_dot_detection__response__pb2.DetectionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'biopb.lacss.Lacss', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('biopb.lacss.Lacss', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class Lacss(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RunDetection(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/biopb.lacss.Lacss/RunDetection',
            lacss_dot_detection__request__pb2.DetectionRequest.SerializeToString,
            lacss_dot_detection__response__pb2.DetectionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RunDetectionStream(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/biopb.lacss.Lacss/RunDetectionStream',
            lacss_dot_detection__request__pb2.DetectionRequest.SerializeToString,
            lacss_dot_detection__response__pb2.DetectionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

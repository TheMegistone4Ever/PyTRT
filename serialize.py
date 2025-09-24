from tensorrt import Logger, Builder, NetworkDefinitionCreationFlag, OnnxParser, MemoryPoolType

TRT_LOGGER = Logger(Logger.WARNING)

builder = Builder(TRT_LOGGER)
network = builder.create_network(1 << int(NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = OnnxParser(network, TRT_LOGGER)

with open("best.onnx", "rb") as f:
    if not parser.parse(f.read()):
        raise RuntimeError(f"ONNX parsing failed:\n{"\n".join(map(parser.get_error, range(parser.num_errors)))}")

config = builder.create_builder_config()
config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, 1 << 30)

serialized_engine = builder.build_serialized_network(network, config)

with open("best.engine", "wb") as f:
    f.write(serialized_engine)

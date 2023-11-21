import orjson

model_id = '{__model_id__}'  # openllm: model id
model_tag = '{__model_tag__}'  # openllm: model tag
adapter_map = orjson.loads("""{__model_adapter_map__}""")  # openllm: model adapter map
serialization = '{__model_serialization__}'  # openllm: model serialization
trust_remote_code = {__model_trust_remote_code__}  # openllm: model trust remote code

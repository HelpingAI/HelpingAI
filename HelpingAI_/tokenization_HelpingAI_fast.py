import json
from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers, processors

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class HelpingAITokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
        add_bos_token=False,
        add_eos_token=False,
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def update_post_processor(self):
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    @property
    def default_chat_template(self):
        return """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

    @property
    def default_tool_template(self):
        return """\\n{%- macro json_to_python_type(json_spec) %}\\n{%- set basic_type_map = {\\n    \"string\": \"str\",\\n    \"number\": \"float\",\\n    \"integer\": \"int\",\\n    \"boolean\": \"bool\"\\n} %}\\n\\n{%- if basic_type_map[json_spec.type] is defined %}\\n    {{- basic_type_map[json_spec.type] }}\\n{%- elif json_spec.type == \"array\" %}\\n    {{- \"List[\" +  json_to_python_type(json_spec.items) + \"]\"}}\\n{%- elif json_spec.type == \"object\" %}\\n    {{- \"Dict[str, \" + json_to_python_type(json_spec.additionalProperties) + ']'}}\\n{%- elif json_spec.type is iterable %}\\n    {{- \"Union[\" }}\\n    {%- for t in json_spec.type %}\\n      {{- json_to_python_type({\"type\": t}) }}\\n      {%- if not loop.last %}\\n        {{- \",\" }} \\n    {%- endif %}\\n    {%- endfor %}\\n    {{- \"]\" }}\\n{%- else %}\\n    {{- \"Any\" }}\\n{%- endif %}\\n{%- endmacro %}\\n\\n{%- macro old_tool_parser(tools) %}\\n{%- for tool in tools %}\\n    {%- if loop.index0 != 0 %}\\n        {{- '\\n\\n' }}\\n    {%- endif %}\\n    {{- '\\npython\\ndef ' + tool.name + '(' }}\\n    {%- for param_name, param_fields in tool.parameter_definitions|items %}\\n        {%- if loop.index0 != 0 %}\\n            {{- ', '}}\\n        {%- endif %}\\n        {{- param_name + ': ' }}\\n        {%- if not param_fields.required %}\\n            {{- 'Optional[' + param_fields.type + '] = None'}}\\n        {%- else %}\\n            {{- param_fields.type }}\\n        {%- endif %}\\n    {%- endfor %}\\n    {{- ') -> List[Dict]:\\n    \"\"\"'}}\\n    {{- tool.description }}\\n    {%- if tool.parameter_definitions|length != 0 %}\\n        {{- '\\n\\n    Args:\\n        '}}\\n        {%- for param_name, param_fields in tool.parameter_definitions|items %}\\n            {%- if loop.index0 != 0 %}\\n                {{- '\\n        ' }}\\n            {%- endif %}\\n            {{- param_name + ' ('}}\\n            {%- if not param_fields.required %}\\n                {{- 'Optional[' + param_fields.type + ']'}}\\n            {%- else %}\\n                {{- param_fields.type }}\\n            {%- endif %}\\n            {{- '): ' + param_fields.description }}\\n        {%- endfor %}\\n    {%- endif %}\\n    {{- '\\n    \"\"\"\\n    pass\\n\\n' }}\\n{%- endfor %}\\n{%- endmacro %}\\n\\n{%- macro new_tool_parser(tools) %}\\n{%- for tool in tools %}\\n  {%- if loop.index0 != 0 %}\\n    {{- '\\n\\n'}}\\n  {%- endif %}\\n  {%- if tool.function is defined %}\\n    {%- set tool = tool.function %}\\n  {%- endif %}\\n  {{-'\\npython\\ndef ' + tool.name + '('}}\\n  {%- for param_name, param_fields in tool.parameters.properties|items %}\\n    {%- if loop.index0 != 0 %}\\n      {{- ', '}}\\n    {%- endif %}\\n    {{-param_name + \": \"}} \\n    {%- if not param_name in tool.parameters.required %}\\n      {{-'Optional[' + json_to_python_type(param_fields) + '] = None'}}\\n    {%- else %}\\n      {{- json_to_python_type(param_fields) }}\\n    {%- endif %}\\n  {%- endfor %}\\n  {{- ') -> List[Dict]:\\n    \"\"\"'}}\\n  {{- tool.description }}\\n  {%- if tool.parameters.properties|length != 0 %}\\n    {{- '\\n\\n    Args:\\n        '}}\\n    {%- for param_name, param_fields in tool.parameters.properties|items %}\\n      {%- if loop.index0 != 0 %}\\n        {{- '\\n        ' }}\\n      {%- endif %}\\n      {{- param_name + ' ('}}\\n      {%- if not param_name in tool.parameters.required %}\\n        {{-'Optional[' + json_to_python_type(param_fields) + ']'}}\\n      {%- else %}\\n        {{- json_to_python_type(param_fields) }}\\n      {%- endif %}\\n      {{- '): ' + param_fields.description }}\\n    {%- endfor %}\\n    {%- endif %}\\n    {{- '\\n    \"\"\"\\n    pass\\n\\n' }}\\n{%- endfor %}\\n{%- endmacro %}\\n\\n{{- bos_token }}\\n{%- if messages[0]['role'] == 'system' %}\\n  {%- set loop_messages = messages[1:] %}\\n  {%- set system_message = messages[0]['content'] %}\\n{%- else %}\\n  {%- set loop_messages = messages %}\\n  {%- set system_message = '## Task and Context\\nYou help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user\\'s needs as best you can, which will be wide-ranging.\\n\\n## Style Guide\\nUnless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.' %}\\n{%- endif %}\\n{{- '<|im_start|>system\\n' }}\\n{{- '# Safety Preamble' }}\\n{{- '\\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}\\n{{- '\\n\\n# System Preamble' }}\\n{{- '\\n## Basic Rules' }}\\n{{- '\\nYou are a powerful Emotionally Intelligent Conversational AI trained by Abhay to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}\\n{{- '\\n\\n# User Preamble' }}\\n{{- '\\n' + system_message }}\\n{{-'\\n\\n## Available Tools\\nHere is a list of tools that you have available to you:\\n\\n'}}\\n{%- set ns = namespace(new_tools=true) %}\\n{%- for tool in tools %}\\n    {%- if tool.parameter_definitions is defined %}\\n        {%- set ns.new_tools = false %}\\n    {%- endif %}\\n{%- endfor %}\\n{%- if ns.new_tools %}\\n    {{- new_tool_parser(tools) }}\\n{%- else %}\\n    {{- old_tool_parser(tools) }}\\n{%- endif %}\\n{{- '<|im_end|>'}}\\n{%- for message in loop_messages %}\\n  {%- set content = message['content'] %}\\n  {%- if message.role == 'user' %}\\n    {{- '<|im_start|>user\\n' + content|trim + '<|im_end|>' }}\\n  {%- elif message.role == 'system' %}\\n    {{- '<|im_start|>system\\n' + content|trim + '<|im_end|>' }}\\n  {%- elif message.role == 'assistant' and message.tool_calls is defined %}\\n    {{- '<|im_start|>assistant\\n' }}\\n    {%- if message.content is defined %}\\n        {{- message.content|trim }}\\n    {%- endif %}\\n    {{- '\\nAction:\\njson\\n[\\n' }}\\n    {%- for tool_call in message.tool_calls %}\\n        {%- if tool_call.function is defined %}\\n            {%- set tool_call = tool_call.function %}\\n        {%- endif %}\\n        {{- '{\\n'|indent(4, first=true) }}\\n        {{- '\"tool_name\": \"'|indent(8, first=true) + tool_call.name + '\",\\n' }}\\n        {{- '\"parameters\": '|indent(8, first=true) }}\\n        {%- if tool_call.arguments is defined and tool_call.arguments|length > 0 %}    \\n            {{- tool_call.arguments|tojson(indent=4)|indent(8) }}\\n            {{- '\\n' }}\\n        {%- else %}\\n            {{- '{}\\n' }}\\n        {%- endif %}\\n        {{- '}'|indent(4, first=true) }}\\n        {%- if not loop.last %}\\n            {{- ',\\n' }}\\n        {%- endif %}\\n    {%- endfor %}\\n    {{- \"\\n]\\n\" }}\\n  {%- elif message.role == 'assistant' %}\\n    {{- '<|im_start|>assistant\\n'  + content|trim + '<|im_end|>' }}\\n  {%- elif message.role == 'tool' %}\\n    {{- '<|im_start|>system\\n<results>\\n' }}\\n    {{- message.content|trim }}\\n    {{- '</results><|im_end|>' }}\\n  {%- endif %}\\n{%- endfor %}\\n{{-'<|im_start|>system\\nWrite \\'Action:\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the directly-answer tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\\njson\\n[\\n    {\\n        \"tool_name\": title of the tool in the specification,\\n        \"parameters\": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\\n    }\\n]\\n<|im_end|>'}}\\n{%- if add_generation_prompt %}\\n  {{- '<|im_start|>assistant\\n' }}\\n{%- endif %}\\n"""

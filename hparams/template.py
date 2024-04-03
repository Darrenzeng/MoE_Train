from dataclasses import dataclass
from transformers import PreTrainedTokenizer,AutoTokenizer
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from formatter import EmptyFormatter, StringFormatter, Formatter

@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_observation: "Formatter"
    format_separator: "Formatter"
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    force_system: bool
    #仅仅实现把输入给包裹起来就行
    pass

templates: Dict[str, Template] = {}

def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    default_system: Optional[str] = "",
    stop_words: Optional[List[str]] = [],
    efficient_eos: Optional[bool] = False,
    replace_eos: Optional[bool] = False,
    force_system: Optional[bool] = False,
) -> None:
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    template_class = Template
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_separator_formatter = EmptyFormatter()
    templates[name] = template_class(#每调用一次该方法，就在templates中加入了一个新的模板
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_separator=format_separator or default_separator_formatter,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
    )

def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    is_oov = eos_token not in tokenizer.get_vocab()
    tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        print("Add eos token: {}".format(tokenizer.eos_token))
    else:
        print("Replace eos token: {}".format(tokenizer.eos_token))


def get_template_and_fix_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    name: Optional[str] = None,
) -> Template:
    if name is None:
        template = templates["vanilla"]  # placeholder
    else:
        template = templates.get(name, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(name))

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        print("Add {} to stop words.".format(",".join(stop_words)))

    return template


_register_template(
    name="qwen",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)

def add_prompt_form_template(template, query):
    system_prompt = template.format_system.apply(content=template.default_system)[0]
    text = template.format_user.apply(content=query)[0]

    return system_prompt + text

if __name__=="__main__":
    test = "好的，没问题"
    tokenizer = AutoTokenizer.from_pretrained("/Users/a58/Downloads/pretrain_model/Qwen/Qwen1.5-0.5B-Chat")
    template = get_template_and_fix_tokenizer(tokenizer, name="qwen")
    query_new = add_prompt_form_template(template=template, query=test)
    print("ok")
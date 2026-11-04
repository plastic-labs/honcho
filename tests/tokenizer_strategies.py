import os

import tiktoken
from hypothesis import settings
from hypothesis import strategies as st

TOKENIZER_PROPERTIES = settings(
    max_examples=int(os.environ.get("HONCHO_FUZZ_EXAMPLES", "25")),
    deadline=None,
)


def special_token_content(encoding_name: str) -> st.SearchStrategy[str]:
    encoding = tiktoken.get_encoding(encoding_name)
    surrounding_text = st.one_of(
        st.text(
            alphabet=st.characters(
                exclude_categories=("Cs",), exclude_characters="\x00"
            ),
            max_size=40,
        ),
        st.sampled_from(
            [
                "",
                " ",
                "\n\t",
                "猫" * 40,
                "\U0001f9d1\U0001f3fd\u200d\U0001f4bb" * 10,
                "<|",
                "|>",
            ]
        ),
    )
    special_tokens = st.lists(
        st.sampled_from(sorted(encoding.special_tokens_set)),
        min_size=1,
        max_size=4,
    ).map("".join)
    return st.tuples(surrounding_text, special_tokens, surrounding_text).map("".join)

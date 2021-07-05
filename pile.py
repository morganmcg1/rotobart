import datasets
import io
import jsonlines
import zstandard as zstd

logger = datasets.logging.get_logger(__name__)
_URLS = {
    datasets.Split.TRAIN: [
        f"https://the-eye.eu/public/AI/pile/train/{str(index).zfill(2)}.jsonl.zst" for index in range(30)
    ],
    datasets.Split.VALIDATION: ["https://the-eye.eu/public/AI/pile/val.jsonl.zst"],
    datasets.Split.TEST: ["https://the-eye.eu/public/AI/pile/test.jsonl.zst"],
}


class Pile(datasets.GeneratorBasedBuilder):
    """The Pile, An 800GB Dataset of Diverse Text for Language Modeling"""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
#                     "id": datasets.Value("int64"),
                    "text": datasets.Value("string")
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=split, gen_kwargs={"filepaths": dl_manager.download(_URLS[split])})
            for split in _URLS
        ]

    def _generate_examples(self, filepaths):
        id_ = 0
        dctx = zstd.ZstdDecompressor()
        for filepath in filepaths:
            print(f"Generate examples from: {filepath}")
            with open(filepath, "rb") as f:
                f = dctx.stream_reader(f)
                f = io.TextIOWrapper(f, encoding="utf-8")
                f = jsonlines.Reader(f)
                for line in f:
#                     sentence = {"text": line["text"], "pile_set_name": line["meta"]["pile_set_name"] }
#                     yield id_, {"id": id_, "text": line["text"]}
                      yield id_, line["text"]
                      id_ += 1

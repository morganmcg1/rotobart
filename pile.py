import io
import os

import datasets
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
                    "id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "pile_set_name": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files is None:
            return [
                datasets.SplitGenerator(name=split, gen_kwargs={"filepaths": dl_manager.download(_URLS[split])})
                for split in _URLS
            ]
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tupe)):
            if isinstance(data_files, str):
                if os.path.isdir(data_files):
                    data_files = [
                        os.path.join(data_files, filename)
                        for filename in os.listdir(data_files)
                        if filename.endswith("zst")
                    ]
                else:
                    data_files = [data_files]
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": data_files})]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                if os.path.isdir(files):
                    files = [
                        os.path.join(files, filename) for filename in os.listdir(files) if filename.endswith("zst")
                    ]
                else:
                    files = [data_files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"filepaths": files}))

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
                    yield id_, {"id": id_, "text": line["text"], "pile_set_name": line["meta"]["pile_set_name"]}
                    id_ += 1

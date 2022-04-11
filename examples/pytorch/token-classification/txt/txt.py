# import logging
# from dataclasses import dataclass
#
# import pyarrow.csv as pac
#
# import datasets
#
#
# logger = logging.getLogger(__name__)
#
# FEATURES = datasets.Features(
#     {
#         "text": datasets.Value("string"),
#     }
# )
#
#
# @dataclass
# class TextConfig(datasets.BuilderConfig):
#     """BuilderConfig for text files."""
#
#     encoding: str = None
#     block_size: int = None
#     use_threads: bool = None
#     read_options: pac.ReadOptions = None
#     parse_options: pac.ParseOptions = None
#     convert_options: pac.ConvertOptions = None
#
#     @property
#     def pa_read_options(self):
#         if self.read_options is not None:
#             read_options = self.read_options
#         else:
#             read_options = pac.ReadOptions(column_names=["text"])
#         if self.encoding is not None:
#             read_options.encoding = self.encoding
#         if self.block_size is not None:
#             read_options.block_size = self.block_size
#         if self.use_threads is not None:
#             read_options.use_threads = self.use_threads
#         return read_options
#
#     @property
#     def pa_parse_options(self):
#         if self.parse_options is not None:
#             parse_options = self.parse_options
#         else:
#             parse_options = pac.ParseOptions(
#                 delimiter="\r",
#                 quote_char=False,
#                 double_quote=False,
#                 escape_char=False,
#                 newlines_in_values=False,
#                 ignore_empty_lines=False,
#             )
#         return parse_options
#
#     @property
#     def pa_convert_options(self):
#         if self.convert_options is not None:
#             convert_options = self.convert_options
#         else:
#             convert_options = pac.ConvertOptions(
#                 column_types=FEATURES.type,
#             )
#         return convert_options
#
#
# class Text(datasets.ArrowBasedBuilder):
#     BUILDER_CONFIG_CLASS = TextConfig
#
#     def _info(self):
#         return datasets.DatasetInfo(features=FEATURES)
#
#     def _split_generators(self, dl_manager):
#         """The `datafiles` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].
#
#         If str or List[str], then the dataset returns only the 'train' split.
#         If dict, then keys should be from the `datasets.Split` enum.
#         """
#         if not self.config.data_files:
#             raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
#         data_files = dl_manager.download_and_extract(self.config.data_files)
#         if isinstance(data_files, (str, list, tuple)):
#             files = data_files
#             if isinstance(files, str):
#                 files = [files]
#             return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})]
#         splits = []
#         for split_name in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]:
#             if split_name in data_files:
#                 files = data_files[split_name]
#                 if isinstance(files, str):
#                     files = [files]
#                 splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
#         return splits
#
#     def _generate_tables(self, files):
#         for i, file in enumerate(files):
#             pa_table = pac.read_csv(
#                 file,
#                 read_options=self.config.pa_read_options,
#                 parse_options=self.config.pa_parse_options,
#                 convert_options=self.config.convert_options,
#             )
#             # Uncomment for debugging (will print the Arrow table size and elements)
#             # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
#             # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
#             yield i, pa_table


# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to MSRA NER Dataset"""

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{levow2006third,
  author    = {Gina{-}Anne Levow},
  title     = {The Third International Chinese Language Processing Bakeoff: Word
               Segmentation and Named Entity Recognition},
  booktitle = {SIGHAN@COLING/ACL},
  pages     = {108--117},
  publisher = {Association for Computational Linguistics},
  year      = {2006}
}
"""

_DESCRIPTION = """\
The Third International Chinese Language
Processing Bakeoff was held in Spring
2006 to assess the state of the art in two
important tasks: word segmentation and
named entity recognition. Twenty-nine
groups submitted result sets in the two
tasks across two tracks and a total of five
corpora. We found strong results in both
tasks as well as continuing challenges.

MSRA NER is one of the provided dataset.
There are three types of NE, PER (person),
ORG (organization) and LOC (location).
The dataset is in the BIO scheme.

For more details see https://faculty.washington.edu/levow/papers/sighan06.pdf
"""

# _URL = "https://raw.githubusercontent.com/OYE93/Chinese-NLP-Corpus/master/NER/MSRA/"
# _TRAINING_FILE = "msra_train_bio.txt"
# _TEST_FILE = "msra_test_bio.txt"


class TextNerConfig(datasets.BuilderConfig):
    """BuilderConfig for MsraNer"""

    def __init__(self, **kwargs):
        """BuilderConfig for MSRA NER.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TextNerConfig, self).__init__(**kwargs)


class TextNer(datasets.GeneratorBasedBuilder):
    """MSRA NER dataset."""

    BUILDER_CONFIGS = [
        TextNerConfig(name="text_ner", version=datasets.Version("1.0.0"), description="TEXT NER dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-ORG",
                                "I-ORG",
                                "B-LOC",
                                "I-LOC",
                                "B-DATE",
                                "I-DATE",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.microsoft.com/en-us/download/details.aspx?id=52531",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # urls_to_download = {
        #     "train": f"{_URL}{_TRAINING_FILE}",
        #     "test": f"{_URL}{_TEST_FILE}",
        # }
        """The `datafiles` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].

                If str or List[str], then the dataset returns only the 'train' split.
                If dict, then keys should be from the `datasets.Split` enum.
                """
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})]
        splits = []
        for split_name in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]:
            if split_name in data_files:
                files = data_files[split_name]
                if isinstance(files, str):
                    files = [files]
                splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        for i, file in enumerate(files):
            logger.info("⏳ Generating examples from = %s", file)
            with open(file, encoding="utf-8") as f:
                guid = 0
                tokens = []
                ner_tags = []
                for line in f:
                    line_stripped = line.strip()
                    if line_stripped == "":
                        if tokens:
                            yield guid, {
                                "id": str(guid),
                                "tokens": tokens,
                                "ner_tags": ner_tags,
                            }
                            guid += 1
                            tokens = []
                            ner_tags = []
                    else:
                        splits = line_stripped.split("\t")
                        if len(splits) == 1:
                            splits.append("O")
                        tokens.append(splits[0])
                        ner_tags.append(splits[1])
                # last example
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }

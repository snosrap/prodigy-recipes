import spacy

from typing import List, Optional, Dict, Any, Union, Iterable
from pathlib import Path
from copy import deepcopy
from itertools import groupby

from prodigy.recipes import textcat
from prodigy.core import recipe
from prodigy.util import get_labels, split_string
from prodigy.models.matcher import PatternMatcher
from prodigy.components.preprocess import split_spans, add_label_options, add_labels_to_stream

# The opposite of split_spans https://prodi.gy/docs/api-components#split_spans
# Take a list of documents (each containing a single span), group them by their _input_hash,
# then take the document annotation fields and add them to each span
def join_spans(answers, fields=['accept','answer']):
    docs = []
    for parent_input_hash, doc_splits in groupby(answers, key=lambda x: x['_input_hash']):
        doc_splits = sorted(doc_splits, key=lambda x:x['spans'][0]['start']) # convert grouper to list and order by the single-span start index
        doc = deepcopy(doc_splits[0]) # initialize with any matching doc_split (e.g., the first one)
        doc['spans'] = [] # empty the spans
        for d in doc_splits: # add document-level annotation fields (e.g., accept, answer) into the appropriate span (already sorted above)
            assert(len(d['spans']) == 1) # if we called split_spans, there should only be one span per document
            span = d['spans'][0] # get the only span
            for field in fields: # move the document annotation fields into the span
                span[field] = d[field]
            doc['spans'].append(span)
        for field in fields: # clean up doc annotations (since they're really span annotations)
            del doc[field]
        docs.append(doc)
    return docs

def update(answers):
    # HACK: re-join the spans and save to a global variable
    # This is necesary because the `update` return value is not stored/sent anywhere (e.g., `before_db`)
    # SEE: https://support.prodi.gy/t/textcat-teach-with-multiple-choice-update-model/2533/2
    global updated_answers 
    updated_answers = join_spans(answers)

def before_db(answers):
    return updated_answers # ignore the `answers` parameter and just return the global `updated_answers` that was just set in `update`

@recipe(
    "nercat.manual",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy model for tokenization or blank:lang (e.g. blank:en)", "positional", None, str), # from ner.manual
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    exclusive=("Treat classes as mutually exclusive (if not set, an example can have multiple correct classes)", "flag", "E", bool),
    exclude=("Comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    patterns=("Path to match patterns file", "option", "pt", str), # from ner.manual
    # fmt: on
)
def manual(
    dataset: str,
    spacy_model: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    exclusive: bool = False,
    exclude: Optional[List[str]] = None,
    patterns: Optional[str] = None,
) -> Dict[str, Any]:
    # override textcat.manual
    obj = textcat.manual(dataset,source,loader,label,exclusive,exclude)

    # create patterns, as in ner.manual.
    # TODO: don't really need to do this if we already have spans/entities
    nlp = spacy.blank(spacy_model.replace("blank:", "")) if spacy_model.startswith("blank:") else spacy.load(spacy_model)
    if patterns is not None:
        pattern_matcher = PatternMatcher(nlp, combine_matches=True, all_examples=True)
        pattern_matcher = pattern_matcher.from_disk(patterns)
        obj['stream'] = (eg for _, eg in pattern_matcher(obj['stream'])) # label the patterns, as in ner.manual

    # split_spans, which retains the _input_hash, but creates a new _task_hash
    obj['stream'] = split_spans(obj['stream']) # splits each document into N copies, one-per-span
    obj['stream'] = add_label_options(obj['stream'], label) # re-add the labels, since split_spans doesn't include them and prodigy will throw a javascript error if the `options` key is missing

    # join_spans before saving to the database
    obj['update'] = update
    obj['before_db'] = before_db

    # auto-accept when selecting an annotation
    obj['config']['choice_auto_accept'] = True

    return obj
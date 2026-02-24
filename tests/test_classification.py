import pandas as pd

from kalshi_longshot_bias.features import classify_structure


def test_single_contract():
    df = pd.DataFrame(
        [
            {"ticker": "A", "event_ticker": "EVT1", "title": "Will it rain?", "subtitle": ""},
        ]
    )
    labels = classify_structure(df)
    assert labels.iloc[0] == "single"


def test_numeric_buckets():
    df = pd.DataFrame(
        [
            {"ticker": "A", "event_ticker": "EVT2", "title": "0-10", "subtitle": ""},
            {"ticker": "B", "event_ticker": "EVT2", "title": "11-20", "subtitle": ""},
            {"ticker": "C", "event_ticker": "EVT2", "title": "21-30", "subtitle": ""},
        ]
    )
    labels = classify_structure(df)
    assert set(labels) == {"numeric_buckets"}


def test_threshold_ladder():
    df = pd.DataFrame(
        [
            {"ticker": "A", "event_ticker": "EVT3", "title": "At least 1", "subtitle": ""},
            {"ticker": "B", "event_ticker": "EVT3", "title": "At least 2", "subtitle": ""},
            {"ticker": "C", "event_ticker": "EVT3", "title": "At least 3", "subtitle": ""},
        ]
    )
    labels = classify_structure(df)
    assert set(labels) == {"ladder"}


def test_mutual_other():
    df = pd.DataFrame(
        [
            {"ticker": "A", "event_ticker": "EVT4", "title": "Alice", "subtitle": ""},
            {"ticker": "B", "event_ticker": "EVT4", "title": "Bob", "subtitle": ""},
            {"ticker": "C", "event_ticker": "EVT4", "title": "Charlie", "subtitle": ""},
        ]
    )
    labels = classify_structure(df)
    assert set(labels) == {"mutual_other"}

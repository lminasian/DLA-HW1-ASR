import torchaudio
import numpy as np

# @lminasian: will filtrate such cases
# TODO: add filtration for empty target text in dataset

def calc_cer(target_text, predicted_text) -> float:
    assert len(target_text) > 0
    return 100 * torchaudio.functional.edit_distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    assert len(target_text) > 0
    return 100 * torchaudio.functional.edit_distance(target_text.split(), predicted_text.split()) / len(target_text.split())


def test_calc_cer():
    inputs = [
        ("Java is the best", "Guava iz the best"),
        ("How", "Bow"),
        ("New Testament", "Old Testament"),
        ("hello", "hellooooo"),
    ]

    targets = [
        3 / len(inputs[0][0]) * 100,
        1 / len(inputs[1][0]) * 100,
        3 / len(inputs[2][0]) * 100,
        4 / len(inputs[3][0]) * 100,
    ]

    for input, target in zip(inputs, targets):
        assert np.isclose(calc_cer(*input), target)
 

def test_calc_wer():
    inputs = [
        ("Java is the best", "Guava iz the best"),
        ("How", "Bow"),
        ("New Testament", "Old Testament"),
        ("hello", "hellooooo"),
    ]

    targets = [
        2 / len(inputs[0][0].split()) * 100,
        1 / len(inputs[1][0].split()) * 100,
        1 / len(inputs[2][0].split()) * 100,
        1 / len(inputs[3][0].split()) * 100,
    ]

    for input, target in zip(inputs, targets):
        assert np.isclose(calc_wer(*input), target)

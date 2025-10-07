import editdistance

# TODO: Don't forget to support cases when target_text == ''

# @lminasian: will filtrate such cases
# TODO: add filtration for empty target text in dataset

def calc_cer(target_text, predicted_text) -> float:
    assert len(target_text) > 0
    return 100 * editdistance.eval(target_text, predicted_text) / len(target_text)
    


def calc_wer(target_text, predicted_text) -> float:
    assert len(target_text) > 0
    return 100 * editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split())

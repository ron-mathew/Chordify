"""
Chordify key/tonality detection.

Adapted from Lamucal modulation.py.
Uses diatonic scoring to identify the most likely tonal centre.
"""
from collections import defaultdict
from itertools import product

from utils.labels import MAJOR_CHORDS, MINOR_CHORDS, MAJ2MIN_MAP


def get_chord_tonic(chord: str) -> str:
    chord = chord.replace("Dim", "dim").replace("Aug", "aug")
    if len(chord) >= 3:
        if chord[1] in ["#", "b"]:
            if "maj7" in chord:
                return chord[:2]
            elif chord[2] in ["m"]:
                return chord[:3]
            elif any(t in chord for t in ["sus2", "sus4", "dim", "aug"]):
                return chord
            else:
                return chord[:2]
        elif chord[1] == "5":
            return chord[:2]
        elif any(t in chord for t in ["sus2", "sus4", "dim", "aug"]):
            return chord
        elif "maj7" in chord:
            return chord.replace("maj7", "")
        elif "7M" in chord:
            return chord.replace("7M", "")
        elif chord[1] in ["m"]:
            return chord[:2]
        else:
            return chord[:1]
    elif len(chord) == 2:
        if chord[1] in ["#", "b", "m", "5"]:
            return chord[:2]
        else:
            return chord[:1]
    else:
        return chord[:1]


def chord_name_filter(chords: list) -> list:
    result = []
    for c in chords:
        if c:
            result.append(get_chord_tonic(c))
    return result


def chord_to_tone(chord: str) -> str:
    tone = chord[0]
    if len(chord) > 2 and chord[1] in ["b", "#"]:
        tone += chord[1]
    return tone


def get_chords_repeat_num3(chords: list) -> dict:
    score_map = {}
    for ex_chords in MAJOR_CHORDS:
        score_map[ex_chords[0]] = len([c for c in chords if c in ex_chords])
    return score_map


def process_sus_chords(filter_total_chords: list, sus_chords: list):
    tone_count_dic = defaultdict(int)
    for chord in sus_chords:
        tone_count_dic[chord] += 1
    keys = list(tone_count_dic)
    product_list = list(product(*([[0, 1]] * len(keys))))
    chords_score_list = []
    for item in product_list:
        sus_conv_chord = []
        for i, x in enumerate(item):
            key = keys[i]
            tone = chord_to_tone(key)
            count = tone_count_dic[key]
            sus_conv_chord.extend(
                [tone] * count if x == 0 else [tone + "m"] * count
            )
        score_map = get_chords_repeat_num3(filter_total_chords + sus_conv_chord)
        chords_score_list.extend(
            [(k, v, item) for k, v in score_map.items() if v != 0]
        )
    chords_score_list.sort(key=lambda x: x[1], reverse=True)
    return chords_score_list, keys


def get_ret_key(best_key, chord, keys, maj_min_dic, is_sus, ret_key, sus_dic):
    for k, score, item in best_key:
        if any(t in chord for t in ["sus", "dim", "aug", "5"]):
            tone = chord_to_tone(chord)
            if item[keys.index(chord)] == 1:
                tone += "m"
            chord = tone
        if chord == maj_min_dic[k][0] or chord == maj_min_dic[k][5]:
            ret_key = k
            if is_sus:
                sus_dic = {
                    keys[i]: chord_to_tone(keys[i]) if v == 0
                    else chord_to_tone(keys[i]) + "m"
                    for i, v in enumerate(item)
                }
            break
    return ret_key, sus_dic


def get_ret_key2(best_key, head_chord, tail_chord, maj_min_dic, ret_key):
    for k, score in best_key:
        if head_chord == maj_min_dic[k][0] or head_chord == maj_min_dic[k][5]:
            ret_key = k
            break
    if not ret_key:
        for k, score in best_key:
            if tail_chord == maj_min_dic[k][0] or tail_chord == maj_min_dic[k][5]:
                ret_key = k
                break
    return ret_key


def process_best_key(chords_score_list, head_chord, tail_chord, keys,
                     maj_min_dic, ret_key, sus_dic, is_sus=True):
    if not chords_score_list:
        return None, None
    if is_sus:
        best_key = [x for x in chords_score_list if x[1] == chords_score_list[0][1]]
    else:
        best_key = [x for x in chords_score_list if x[-1] == chords_score_list[0][-1]]

    if len(best_key) > 1 and is_sus:
        ret_key, sus_dic = get_ret_key(
            best_key, head_chord, keys, maj_min_dic, is_sus, ret_key, sus_dic
        )
        if not ret_key:
            ret_key, sus_dic = get_ret_key(
                best_key, tail_chord, keys, maj_min_dic, is_sus, ret_key, sus_dic
            )
    elif len(best_key) > 1 and not is_sus:
        ret_key = get_ret_key2(best_key, head_chord, tail_chord, maj_min_dic, ret_key)

    if not ret_key:
        ret_key = best_key[0][0]
        if is_sus and len(best_key[0]) >= 3:
            sus_dic = {
                keys[i]: chord_to_tone(keys[i]) if v == 0
                else chord_to_tone(keys[i]) + "m"
                for i, v in enumerate(best_key[0][2])
            }
    return ret_key, sus_dic


def get_maj_or_min_key(ret_key, maj_min_dic, head_chord, tail_chord, sus_dic):
    maj_key = ret_key
    min_key = MAJ2MIN_MAP.get(ret_key, ret_key + "m")
    maj_chords = maj_min_dic[maj_key]

    def resolve_sus(chord):
        if any(t in chord for t in ["sus", "dim", "aug", "5"]):
            return sus_dic.get(chord, chord)
        return chord

    head_chord = resolve_sus(head_chord)
    if head_chord == maj_chords[0]:
        return maj_key
    if head_chord == maj_chords[5]:
        return min_key

    tail_chord = resolve_sus(tail_chord)
    if tail_chord == maj_chords[0]:
        return maj_key
    if tail_chord == maj_chords[5]:
        return min_key
    return maj_key


def search_key(total_chords: list) -> tuple[str, int]:
    """
    Estimate the tonal key from a list of chord names.

    Returns
    -------
    (key_string, diatonic_score)
    """
    if not total_chords:
        return "C", 0

    maj_min_dic = {x[0]: x for x in MAJOR_CHORDS + MINOR_CHORDS}
    ret_key = None
    keys: list = []
    sus_dic: dict = {}

    total_chords = chord_name_filter(total_chords)
    if not total_chords:
        return "C", 0

    head_chord = total_chords[0]
    tail_chord = total_chords[-1]

    sus_chords = [
        c for c in total_chords
        if any(t in c for t in ["sus", "dim", "aug", "5"])
    ]
    filter_total_chords = [
        c for c in total_chords
        if not any(t in c for t in ["sus", "dim", "aug", "5"])
    ]

    if not filter_total_chords and not sus_chords:
        return "C", 0

    if sus_chords:
        chords_score_list, keys = process_sus_chords(filter_total_chords, sus_chords)
        ret_key, sus_dic = process_best_key(
            chords_score_list, head_chord, tail_chord, keys,
            maj_min_dic, ret_key, sus_dic
        )
    else:
        score_map = get_chords_repeat_num3(filter_total_chords)
        chords_score_list = sorted(
            [(k, v) for k, v in score_map.items() if v != 0],
            key=lambda x: x[1],
            reverse=True,
        )
        ret_key, sus_dic = process_best_key(
            chords_score_list, head_chord, tail_chord, keys,
            maj_min_dic, ret_key, sus_dic, is_sus=False
        )

    if not ret_key:
        return "C", 0

    key_score = 0
    for item in chords_score_list:
        if item[0] == ret_key:
            key_score = item[1]
            break

    ret_key = get_maj_or_min_key(ret_key, maj_min_dic, head_chord, tail_chord, sus_dic)
    return ret_key, key_score

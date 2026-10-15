"""The scene lists bundled with the NuRec parser."""

from py123d.parser.nurec.nurec_splits import (
    NUREC_DEFAULT_SPLITS,
    load_default_scene_lists,
    scenes_of_split,
    validate_splits,
)


def test_bundled_splits():
    scene_lists = load_default_scene_lists()

    assert {split: len(scenes) for split, scenes in scene_lists.items()} == {
        "nurec-curated_train": 1761,
        "nurec-curated_val": 441,
        "nurec-other_train": 162,
    }
    assert NUREC_DEFAULT_SPLITS == ("nurec-curated_train", "nurec-curated_val")
    assert validate_splits(["nurec-other_train"], scene_lists) == ["nurec-other_train"]


def test_other_split_shares_no_scene_with_the_curated_splits():
    scene_lists = load_default_scene_lists()
    uuids = {split: {uuid for _, uuid in scenes_of_split(split, scene_lists)} for split in scene_lists}

    assert not uuids["nurec-other_train"] & uuids["nurec-curated_train"]
    assert not uuids["nurec-other_train"] & uuids["nurec-curated_val"]
    assert not uuids["nurec-curated_train"] & uuids["nurec-curated_val"]

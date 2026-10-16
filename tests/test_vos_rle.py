import numpy as np
import pytest

from src.vos import rle


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roundtrip_uncompressed(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((37, 53)) > 0.6
    assert np.array_equal(rle.decode(rle.encode(m)), m)


def test_starts_with_zero_run_when_first_pixel_is_set():
    m = np.ones((2, 2), bool)
    enc = rle.encode(m)
    assert enc["counts"][0] == 0
    assert np.array_equal(rle.decode(enc), m)


def test_roundtrip_compressed_string():
    rng = np.random.default_rng(3)
    m = np.zeros((120, 90), bool)
    m[10:80, 20:70] = True
    m[rng.random(m.shape) > 0.97] ^= True
    enc = rle.encode_compressed(m)
    assert isinstance(enc["counts"], str)
    assert np.array_equal(rle.decode(enc), m)
    assert rle.area(enc) == int(m.sum())


def test_matches_pycocotools_if_available():
    mask_api = pytest.importorskip("pycocotools.mask")
    rng = np.random.default_rng(7)
    m = rng.random((64, 48)) > 0.5
    ref = mask_api.encode(np.asfortranarray(m.astype(np.uint8)))
    ours = rle.encode_compressed(m)
    assert ours["counts"] == ref["counts"].decode("ascii")
    assert np.array_equal(rle.decode(ref), m)


def test_rejects_inconsistent_counts():
    with pytest.raises(ValueError):
        rle.decode({"size": [2, 2], "counts": [1, 1]})

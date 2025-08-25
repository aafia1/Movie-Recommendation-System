from src.utils import precision_at_k

def test_precision_at_k_basic():
    recs = [1,2,3,4,5]
    rel = {2,5,7}
    assert abs(precision_at_k(recs, rel, k=3) - 1/3) < 1e-9
    assert abs(precision_at_k(recs, rel, k=5) - 2/5) < 1e-9

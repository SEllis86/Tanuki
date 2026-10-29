"""Folder-name labels and the donor-level volcano statistics. No data needed."""
import numpy as np

from tanuki.utils import cell_type_of, mz_col_name, slide_id_of
from tanuki.volcano import bh, moderated_one_sample


def test_labels_from_folder_names():
    assert cell_type_of("A1_micro_11291_S2B1") == "MICRO"
    assert cell_type_of("B2_Neurons_11302_S3B1") == "NEURO"
    assert cell_type_of("A2_astro_11291_stim_S2B2") == "ASTRO"
    assert slide_id_of("A1_1_neuro_11450_S2B1") == "S2B1"
    assert slide_id_of("A1_micro_11291") == "UnknownSlide"
    assert mz_col_name(760.58512) == "mz_760.5851"


def test_moderated_t_holds_its_size_under_the_null():
    rng = np.random.default_rng(0)
    D = rng.normal(0, rng.uniform(0.3, 1.5, size=(1, 2000)), size=(3, 2000))   # 3 donors x 2000 m/z, no effect
    mean, t, p, df_tot, d0 = moderated_one_sample(D)
    assert p.shape == (2000,) and np.all((p >= 0) & (p <= 1))
    assert 0.02 < np.mean(p < 0.05) < 0.08
    assert np.nanmin(bh(p)) > 0.05


def test_bh_is_monotone():
    p = np.array([0.001, 0.01, 0.02, 0.5, np.nan])
    q = bh(p)
    assert np.isnan(q[-1]) and np.all(np.diff(q[:-1]) >= 0) and np.all(q[:-1] >= p[:-1])

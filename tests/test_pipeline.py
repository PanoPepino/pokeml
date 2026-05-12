# scripts/smoke_test_pipeline.py

import pandas as pd

from pokeml.pipeline.prepare import prepare_data_train
from pokeml.classifier.trainer import BandClassifier


def main():
    # tiny sample for fast pipeline validation
    data, fe_state = prepare_data_train("../datasets/pkdx_min.csv", tsize=0.2)

    assert set(data.keys()) == {"cat_native", "cat_ordinal", "light_gbm"}

    X_tr_native, X_te_native, y_tr_native, y_te_native, cats_native = data["cat_native"]

    # reduce size to keep smoke test fast
    X_tr_native = X_tr_native.head(80).copy()
    y_tr_native = y_tr_native.head(80).copy()
    X_te_native = X_te_native.head(20).copy()

    label_helper_df = X_tr_native.copy()
    label_helper_df["total_stats"] = y_tr_native.values

    classifier = BandClassifier(iterations=30, verbose=0)
    classifier.fit(
        X=X_tr_native,
        y_num=y_tr_native,
        cat_features=cats_native,
        full_df_for_labels=label_helper_df,
    )

    X_enriched = classifier.enrich(X_te_native)

    print(X_enriched.head(10))

    assert "pred_band" in X_enriched.columns
    assert any(col.startswith("proba_") for col in X_enriched.columns)

    print("✓ smoke test passed")


if __name__ == "__main__":
    main()

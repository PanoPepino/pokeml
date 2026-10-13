"""Fit/replay feature state independently of splitting, encoding and model fitting."""

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


class FeatureEngine:
    """Run original (name, compute, apply) steps without changing their public contract.

    Target-derived features are cross-fitted for training rows. Full training state
    is retained only for validation and future prediction. Group identifiers can
    keep related species together when that metadata becomes available.
    """

    def __init__(self, steps=None):
        self.steps = list(steps or [])
        self.state = {}

    def fit(self, frame):
        work = frame.copy()
        self.state = {}
        for name, compute, apply in self.steps:
            state = compute(work)
            self.state[name] = state
            work = apply(work, state)
        return self

    def transform(self, frame):
        work = frame.copy()
        for name, _, apply in self.steps:
            if name not in self.state:
                raise ValueError(f"Missing fitted state for feature step: {name}")
            work = apply(work, self.state[name])
        return work

    def fit_transform(self, frame, *, groups=None, n_splits=5, random_state=1):
        self.fit(frame)
        if not any(name == "type_deviation" for name, _, _ in self.steps):
            return self.transform(frame)
        count = min(n_splits, len(frame) if groups is None else len(pd.unique(groups)))
        if count < 2:
            raise ValueError("Target-derived features need at least two training folds")
        splitter = (KFold(count, shuffle=True, random_state=random_state)
                    if groups is None else GroupKFold(count))
        pieces = []
        for train_idx, valid_idx in splitter.split(frame, groups=groups):
            engine = FeatureEngine(self.steps).fit(frame.iloc[train_idx])
            piece = engine.transform(frame.iloc[valid_idx])
            piece.index = valid_idx
            pieces.append(piece)
        result = pd.concat(pieces).sort_index()
        result.index = frame.index.copy()
        return result

    @property
    def categorical_features(self):
        return list(dict.fromkeys(c for s in self.state.values()
                                  for c in s.get("new_cat_feats", [])))

    @property
    def maps(self):
        result = {}
        for state in self.state.values():
            result.update(state.get("new_maps") or {})
        return result

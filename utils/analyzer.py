import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import copy
import heapq
from collections import defaultdict
from typing import Callable, Dict, List, Optional


class Analyzer:
    def __init__(
        self,
        loader_fn: Callable,
        queries: List[Dict],
        top_k: int = 500,
        compute_filter_rates: bool = True,
        compute_relevances: bool = True,
        tags_denominator: str = "occurrences",
    ):
        if tags_denominator not in {"occurrences", "rows"}:
            raise ValueError("tags_denominator must be 'occurrences' or 'rows'")

        self.loader_fn = loader_fn

        # Normalize numpy scalars inside queries
        self.queries = [self._clean_query_values(q) for q in copy.deepcopy(queries)]
        self.modified_queries = copy.deepcopy(self.queries)

        self.top_k = int(top_k)
        self.tags_denominator = tags_denominator

        self.filter_fields = self._collect_filter_fields()
        self.relevance_targets = self._collect_relevance_targets()

        self.filter_rates: Dict[str, pd.DataFrame] = {}
        self.relevances: Dict[str, pd.DataFrame] = {}

        if compute_filter_rates and self.filter_fields:
            self._compute_filter_rates()
        if compute_relevances and self.relevance_targets:
            self._compute_relevances()
        
        print(self.filter_rates)
        print(self.relevances)

    # ------------------- normalize numpy scalars -------------------

    def _normalize_value(self, v):
        if isinstance(v, np.generic):
            return v.item()
        return v

    def _clean_query_values(self, q):
        if "scalar_filters" not in q:
            return q
        for sf in q["scalar_filters"]:
            if "value" in sf:
                sf["value"] = self._normalize_value(sf["value"])
        return q


    # ------------------------- helpers ----------------------------

    def _collect_filter_fields(self) -> List[str]:
        fields = set()
        for q in self.queries:
            for f in q.get("scalar_filters", []):
                fields.add(f["field"])
        return list(fields)

    def _collect_relevance_targets(self) -> Dict[str, tuple]:
        valid_ops = {"==", ">", ">=", "<", "<="}
        targets = {}
        for q in self.queries:
            sfs = q.get("scalar_filters", [])
            if len(sfs) != 1:
                continue
            sf = sfs[0]
            if sf["operator"] in valid_ops:
                targets[q["name"]] = (
                    q["reference_vector_name"],
                    q["vector_field"],
                    sf["field"],
                )
        return targets


    # ----------------------- filter rates --------------------------

    def _compute_filter_rates(self):
        counts = {f: defaultdict(int) for f in self.filter_fields}
        total_rows = 0
        total_tag_occurrences = 0

        for df, _, _ in tqdm(self.loader_fn(), desc="Compute filter rates"):
            total_rows += len(df)
            for f in self.filter_fields:
                if f not in df.columns:
                    continue
                if f == "tags":
                    for entry in df["tags"]:
                        if entry is None:
                            continue
                        for v in entry:
                            counts[f][self._normalize_value(v)] += 1
                            total_tag_occurrences += 1
                else:
                    for v in df[f]:
                        counts[f][self._normalize_value(v)] += 1

        self.filter_rates = {}
        for f, cnt in counts.items():
            if not cnt:
                self.filter_rates[f] = pd.DataFrame(columns=["value", "selectivity"])
                continue
            values = list(cnt.keys())
            freqs = [cnt[v] for v in values]

            if f == "tags":
                if self.tags_denominator == "occurrences":
                    denom = max(total_tag_occurrences, 1)
                else:
                    denom = max(total_rows, 1)
            else:
                denom = max(total_rows, 1)

            selectivities = [freq / denom for freq in freqs]
            df_out = pd.DataFrame({"value": values, "selectivity": selectivities})

            try:
                df_out = df_out.sort_values("value").reset_index(drop=True)
            except Exception:
                df_out = df_out.reset_index(drop=True)

            self.filter_rates[f] = df_out


    # ----------------------- relevances ----------------------------

    def _compute_relevances(self):
        ref_vectors = {name: None for name in self.relevance_targets}
        remaining = set(ref_vectors.keys())

        for df, _, _ in tqdm(self.loader_fn(), desc="Locate reference vectors"):
            if not remaining:
                break
            id_map = {rid: idx for idx, rid in enumerate(df["id"])}
            for qname in list(remaining):
                ref_id, vfield, sfield = self.relevance_targets[qname]
                if ref_id in id_map:
                    vec = df[vfield].iloc[id_map[ref_id]]
                    ref_vectors[qname] = np.asarray(vec, dtype="float32")
                    remaining.remove(qname)

        if remaining:
            print(f"[Warning] reference vector not found for queries: {sorted(list(remaining))}")

        grouped_by_vf = defaultdict(list)
        for qname, (_, vfield, sfield) in self.relevance_targets.items():
            if ref_vectors[qname] is not None:
                grouped_by_vf[vfield].append((qname, sfield))

        heaps = {qname: [] for qname in self.relevance_targets.keys()}

        for df, _, _ in tqdm(self.loader_fn(), desc="Compute relevances"):
            for vfield, qlist in grouped_by_vf.items():
                if vfield not in df.columns:
                    continue
                try:
                    vecs = np.stack(df[vfield].values).astype("float32")
                except Exception:
                    continue

                for qname, sfield in qlist:
                    if sfield not in df.columns:
                        continue
                    ref_vec = ref_vectors[qname]
                    if ref_vec is None:
                        continue

                    dists = np.linalg.norm(vecs - ref_vec, axis=1)
                    scalars = df[sfield].values
                    heap = heaps[qname]

                    for dist, sval in zip(dists, scalars):
                        neg = -float(dist)
                        sval = self._normalize_value(sval)
                        if len(heap) < self.top_k:
                            heapq.heappush(heap, (neg, sval))
                        else:
                            if neg > heap[0][0]:
                                heapq.heapreplace(heap, (neg, sval))

        self.relevances = {}
        for qname, heap in heaps.items():
            if not heap:
                continue

            sorted_items = sorted(heap, key=lambda x: -x[0])
            scalar_vals = [self._normalize_value(item[1]) for item in sorted_items]

            vc = (
                pd.Series(scalar_vals)
                .value_counts(normalize=True)
                .reset_index()
                .rename(columns={"index": "value", 0: "selectivity"})
            )

            try:
                vc = vc.sort_values("value").reset_index(drop=True)
            except Exception:
                vc = vc.reset_index(drop=True)

            self.relevances[qname] = vc

    def modify_by_filter_rate(self, query_name: str, target_rate: float):
        q = next((x for x in self.modified_queries if x["name"] == query_name), None)
        if q is None:
            return

        sfs = q.get("scalar_filters", [])
        if not sfs:
            return

        new_filters = []
        for sf in sfs:
            field = sf["field"]
            op = sf["operator"]
            dist = self.filter_rates.get(field)
            if dist is None or dist.empty:
                new_filters.append(sf)
                continue

            try:
                dist_sorted = dist.sort_values("value").reset_index(drop=True)
            except Exception:
                dist_sorted = dist

            cum = 0.0
            chosen = []
            for _, row in dist_sorted.iterrows():
                v = self._normalize_value(row["value"])
                chosen.append(v)
                cum += float(row["selectivity"])
                if cum >= target_rate:
                    break

            if op in {"==", "contains"}:
                for v in chosen:
                    new_filters.append({
                        "field": field, "operator": op, "value": float(v), "logic": "or"
                    })
            else:
                if chosen:
                    if op in {">", ">="}:
                        cutoff = float(chosen[0])
                        new_filters.append({
                            "field": field, "operator": ">=", "value": cutoff, "logic": "and"
                        })
                    else:
                        cutoff = float(chosen[-1])
                        new_filters.append({
                            "field": field, "operator": "<=", "value": cutoff, "logic": "and"
                        })
                else:
                    new_filters.append(sf)

        q["scalar_filters"] = new_filters

    def modify_by_relevance(self, query_name: str, target_rate: float):
        dist = self.relevances.get(query_name)
        if dist is None or dist.empty:
            return

        q = next((x for x in self.modified_queries if x["name"] == query_name), None)
        if q is None:
            return

        sfs = q.get("scalar_filters", [])
        if not sfs:
            return

        chosen = []
        cum = 0.0
        try:
            dist_sorted = dist.sort_values("value").reset_index(drop=True)
        except Exception:
            dist_sorted = dist

        for _, row in dist_sorted.iterrows():
            v = self._normalize_value(row["value"])
            chosen.append(v)
            cum += float(row["proportion"])
            if cum >= target_rate:
                break

        new_filters = []
        for sf in sfs:
            field = sf["field"]
            op = sf["operator"]

            if op in {"==", "contains"}:
                for v in chosen:
                    new_filters.append({
                        "field": field, "operator": op, "value": float(v), "logic": "or"
                    })
            else:
                if chosen:
                    cutoff = float(chosen[-1])
                    if op in {">", ">="}:
                        new_filters.append({
                            "field": field, "operator": ">=", "value": cutoff, "logic": "and"
                        })
                    else:
                        new_filters.append({
                            "field": field, "operator": "<=", "value": cutoff, "logic": "and"
                        })
                else:
                    new_filters.append(sf)

        q["scalar_filters"] = new_filters


    # -------------------------- batch APIs -------------------------

    def bulk_modify(
        self,
        filter_targets: Optional[Dict[str, float]] = None,
        relevance_targets: Optional[Dict[str, float]] = None,
        mode: str = "filter",
    ):
        """
        mode:
            "filter"    -> only modify filter rates
            "relevance" -> only modify relevances
        """
        if mode not in {"filter", "relevance"}:
            raise ValueError("mode must be 'filter' or 'relevance'")

        filter_targets = filter_targets or {}
        relevance_targets = relevance_targets or {}

        if mode == "filter":
            for qname, rate in filter_targets.items():
                try:
                    self.modify_by_filter_rate(qname, float(rate))
                except Exception as e:
                    print(f"[Error] modify_by_filter_rate {qname}: {e}")

        elif mode == "relevance":
            for qname, rate in relevance_targets.items():
                try:
                    self.modify_by_relevance(qname, float(rate))
                except Exception as e:
                    print(f"[Error] modify_by_relevance {qname}: {e}")


    def run(
        self,
        mode: str = "filter",
        default_filter_rate: Optional[float] = None,
        default_relevance_rate: Optional[float] = None,
        per_query_filter: Optional[Dict[str, float]] = None,
        per_query_relevance: Optional[Dict[str, float]] = None,
    ):
        """
        mode:
            "filter"    -> modify only filter rates
            "relevance" -> modify only relevances
        """
        if mode not in {"filter", "relevance"}:
            raise ValueError("mode must be 'filter' or 'relevance'")

        per_query_filter = per_query_filter or {}
        per_query_relevance = per_query_relevance or {}

        f_targets = {}
        r_targets = {}

        for q in self.modified_queries:
            name = q["name"]

            if mode == "filter":
                if name in per_query_filter:
                    f_targets[name] = per_query_filter[name]
                elif default_filter_rate is not None:
                    f_targets[name] = default_filter_rate

            elif mode == "relevance":
                if name in per_query_relevance:
                    r_targets[name] = per_query_relevance[name]
                elif default_relevance_rate is not None:
                    r_targets[name] = default_relevance_rate

        self.bulk_modify(
            filter_targets=f_targets,
            relevance_targets=r_targets,
            mode=mode,
        )


    # ---------------------------- export ---------------------------

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.modified_queries, f, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> Dict:
        return {
            "filter_rates": self.filter_rates,
            "relevances": self.relevances,
            "queries": self.modified_queries,
        }

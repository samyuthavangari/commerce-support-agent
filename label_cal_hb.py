"""Human labels for CAL-100 and HELDOUT-B-50 (frozen 7-rule definition).

Labelled row-by-row in one session, blind to rule output.
Rule IDs: 1 legal · 2 fraud/security · 3 safety · 4 public PII/order-ID ·
5 repeat-contact failure · 6 explicit human-ask · 7 uninterpretable.
"""

CAL_TRUE = [
    # id, rule
    ("CAL_0007", 4), ("CAL_0010", 5), ("CAL_0012", 5), ("CAL_0014", 5),
    ("CAL_0021", 5), ("CAL_0024", 5), ("CAL_0028", 5), ("CAL_0029", 5),
    ("CAL_0031", 5), ("CAL_0037", 7), ("CAL_0038", 5), ("CAL_0039", 5),
    ("CAL_0043", 5), ("CAL_0044", 4), ("CAL_0048", 2), ("CAL_0049", 5),
    ("CAL_0052", 7), ("CAL_0056", 7), ("CAL_0059", 2), ("CAL_0060", 4),
    ("CAL_0064", 5), ("CAL_0065", 5), ("CAL_0069", 4), ("CAL_0070", 5),
    ("CAL_0075", 7), ("CAL_0076", 5), ("CAL_0077", 7), ("CAL_0078", 5),
    ("CAL_0079", 2), ("CAL_0085", 4), ("CAL_0093", 5),
]

HB_TRUE = [
    ("HB_0005", 5), ("HB_0022", 4), ("HB_0023", 2), ("HB_0024", 5),
    ("HB_0028", 5), ("HB_0031", 5), ("HB_0035", 5), ("HB_0036", 4),
    ("HB_0041", 5), ("HB_0047", 2), ("HB_0050", 5),
]


def build_labels(texts_csv, true_list, out_csv, id_prefix):
    import pandas as pd
    df = pd.read_csv(texts_csv)
    true_map = dict(true_list)
    df["escalate_yn"] = df["message_id"].map(lambda m: m in true_map)
    df["escalate_rule"] = df["message_id"].map(lambda m: true_map.get(m, ""))
    df["labelled_by"] = "human_cal_session_v1"
    df["human_verified"] = True
    df.to_csv(out_csv, index=False)
    n = int(df["escalate_yn"].sum())
    print(f"{out_csv}: {len(df)} rows, {n} positives ({n/len(df):.2f})")


if __name__ == "__main__":
    build_labels("golden_set/cal_texts.csv", CAL_TRUE,
                 "golden_set/cal_100.csv", "CAL")
    build_labels("golden_set/hb_texts.csv", HB_TRUE,
                 "golden_set/heldoutB_50.csv", "HB")

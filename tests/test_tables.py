from nacaarchive.pages import Word
from nacaarchive.tables import columns, rows_of, table_after


def W(t, x, y, w=None):
    w = w or 18 * len(t)
    return Word(t, x, y, x + w, y + 30)


def test_table_rows_and_columns():
    cap = dict(kind="table", label="I", text="TABLE I.-PEAK EFFICIENCIES", y0=100, y1=130, x0=300, x1=900)
    words = [W("Blade", 200, 200), W("angle", 300, 200), W("Peak", 700, 200), W("V/nD", 1100, 200)]
    for i, (a, b, c) in enumerate([("15°", "0.81", "0.62"), ("20°", "0.84", "0.81"), ("25°", "0.85", "1.02")]):
        y = 250 + 50 * i
        words += [W(a, 200, y), W(b, 700, y), W(c, 1100, y)]
    words.append(W("The", 200, 700))            # body text after a gap: not in the table
    t = table_after(words, cap)
    assert t["ncols"] == 3 and t["status"] == "auto"
    assert t["rows"][0] == ["Blade angle", "Peak", "V/nD"]
    assert t["rows"][3] == ["25°", "0.85", "1.02"] and len(t["rows"]) == 4


def test_rows_of_groups_by_height():
    rs = rows_of([W("a", 0, 100), W("b", 100, 104), W("c", 0, 150)])
    assert [[w.text for w in r] for r in rs] == [["a", "b"], ["c"]]

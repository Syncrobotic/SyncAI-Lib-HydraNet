"""The 22-class retail taxonomy, as annotated in CVAT task 32.

This is the finest-grained surface scheme in the repo. It exists because the 6-class
`terrain` head cannot answer the questions Stage 0 actually asks of it: a single
`fixture` class merges the display table, the display case and the checkout counter,
and `wall` swallows glass, glass doors and columns -- the three that matter most when
fitting a floor plane, because glass reads as an open path and a column is what an
occluded person is standing behind.

The class names and colours are taken verbatim from the CVAT project's `labelmap.txt`,
so a mask exported from CVAT maps here without an intermediate rename. `background` in
that file is this scheme's `void`, id 0, which the head ignores.

WHAT THIS SCHEME DOES NOT PROMISE
Six of the 22 classes have no public dataset behind them at all -- `平板`, `喇叭`,
`海報`, `消防`, `玻璃門`, `展示櫃` -- and COCO-Stuff supplies no `柱子` either. They can
only be learnt from in-house annotation. Declaring a class the data cannot teach does
not error: the head emits near-zero for it and the class quietly leaves the mIoU
denominator. Check `IoU/terrain/<class>` per class before trusting any average.
"""

from __future__ import annotations

# id 0 is void/background and is mapped to ignore, matching every other scheme here.
RETAIL22 = {
    "void": 0,
    "地板": 1,
    "牆壁": 2,
    "天花板": 3,
    "柱子": 4,
    "玻璃": 5,
    "玻璃門": 6,
    "門": 7,
    "樓梯": 8,
    "展示櫃": 9,
    "展示桌": 10,
    "結帳櫃檯": 11,
    "椅子": 12,
    "盒裝物品": 13,
    "紙箱": 14,
    "筆電": 15,
    "手機": 16,
    "平板": 17,
    "螢幕": 18,
    "喇叭": 19,
    "海報": 20,
    "消防": 21,
    "人": 22,
}

# The class no other taxonomy here names. `utils/visualize.py` keys the retail22 palette
# on it and imports the name from here, so the palette and the taxonomy cannot drift.
GLASS_DOOR = "玻璃門"

# RGB exactly as CVAT writes it in labelmap.txt. A colour that is not listed becomes
# ignore rather than a class, so a palette drift shows up as vanishing supervision
# rather than as silently relabelled pixels.
RETAIL22_COLOR_TO_ID = {
    (0, 0, 0): RETAIL22["void"],
    (104, 196, 108): RETAIL22["地板"],
    (88, 172, 234): RETAIL22["牆壁"],
    (199, 215, 233): RETAIL22["天花板"],
    (103, 204, 119): RETAIL22["柱子"],
    (50, 214, 224): RETAIL22["玻璃"],
    (140, 242, 78): RETAIL22["玻璃門"],
    (78, 242, 160): RETAIL22["門"],
    (161, 136, 127): RETAIL22["樓梯"],
    (245, 124, 65): RETAIL22["展示櫃"],
    (40, 190, 155): RETAIL22["展示桌"],
    (242, 201, 78): RETAIL22["結帳櫃檯"],
    (78, 181, 242): RETAIL22["椅子"],
    (255, 189, 65): RETAIL22["盒裝物品"],
    (242, 167, 78): RETAIL22["紙箱"],
    (78, 126, 242): RETAIL22["筆電"],
    (174, 242, 78): RETAIL22["手機"],
    (248, 92, 122): RETAIL22["平板"],
    (255, 110, 60): RETAIL22["螢幕"],
    (119, 78, 242): RETAIL22["喇叭"],
    (195, 102, 241): RETAIL22["海報"],
    (242, 78, 133): RETAIL22["消防"],
    (236, 64, 122): RETAIL22["人"],
}

# Traversability, in the same 0=blocked / 1=caution / 2=go / 255=ignore vocabulary the
# indoor scheme uses. Everything a person can stand on is `go`; everything that stops
# them or that they would collide with is `blocked`. Glass is blocked and is the most
# dangerous class in the set, because to a depth model it reads as an open path.
RETAIL22_TO_TRAV = {
    0: 255,  # void
    1: 2,  # 地板
    2: 0,  # 牆壁
    3: 255,  # 天花板 — overhead, never underfoot; not a traversability statement
    4: 0,  # 柱子
    5: 0,  # 玻璃 — reads as open path; the reason this class is worth separating
    6: 0,  # 玻璃門
    7: 0,  # 門 — blocked when shut; the opening becomes floor once ajar
    8: 1,  # 樓梯
    9: 0,  # 展示櫃
    10: 0,  # 展示桌
    11: 0,  # 結帳櫃檯
    12: 0,  # 椅子
    13: 0,  # 盒裝物品
    14: 0,  # 紙箱
    15: 0,  # 筆電
    16: 0,  # 手機
    17: 0,  # 平板
    18: 0,  # 螢幕
    19: 0,  # 喇叭
    20: 0,  # 海報
    21: 0,  # 消防
    22: 0,  # 人
}

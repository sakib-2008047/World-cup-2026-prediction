"""Official 2026 World Cup bracket structure (verified June 2026).

Sources: FIFA match schedule via ESPN/Sky/MLSSoccer/Bleacher Report listings.
Slot notation: 'A1' = Group A winner, 'B2' = Group B runner-up,
'3rd:ABCDF' = a third-placed team from one of those groups.

THIRD-PLACE ALLOCATION NOTE: FIFA's Annex C pre-assigns thirds to slots via a
table of 495 scenarios (one per combination of 8 qualifying groups). We do not
hard-code 495 rows; instead we solve the identical constraint problem at
runtime - each third must go to a slot whose pool contains its group, each
slot gets exactly one third. Annex C is one valid solution of this matching;
ours may differ in slot order for some scenarios, which is immaterial for
aggregate Monte Carlo statistics but worth stating in the README.
"""

GROUPS = list("ABCDEFGHIJKL")

#: Round of 32 - match number -> (slot_a, slot_b)
ROUND_OF_32 = {
    73: ("A2", "B2"),
    74: ("E1", "3rd:ABCDF"),
    75: ("F1", "C2"),
    76: ("C1", "F2"),
    77: ("I1", "3rd:CDFGH"),
    78: ("E2", "I2"),
    79: ("A1", "3rd:CEFHI"),
    80: ("L1", "3rd:EHIJK"),
    81: ("D1", "3rd:BEFIJ"),
    82: ("G1", "3rd:AEHIJ"),
    83: ("K2", "L2"),
    84: ("H1", "J2"),
    85: ("B1", "3rd:EFGIJ"),
    86: ("J1", "H2"),
    87: ("K1", "3rd:DEIJL"),
    88: ("D2", "G2"),
}

#: Later rounds - match number -> (feeder_match_a, feeder_match_b)
ROUND_OF_16 = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
}
QUARTERFINALS = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SEMIFINALS = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}

KNOCKOUT_ROUNDS = [ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS, FINAL]
ROUND_NAMES = ["R32", "R16", "QF", "SF", "F"]

#: the eight third-place slots and their allowed group pools
THIRD_SLOTS = {m: set(b.split(":")[1]) for m, (_, b) in ROUND_OF_32.items()
               if b.startswith("3rd:")}

HOST_COUNTRIES = {"United States", "Mexico", "Canada"}

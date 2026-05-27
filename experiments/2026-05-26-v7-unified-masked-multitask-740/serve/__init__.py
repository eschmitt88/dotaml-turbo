"""Downstream queries on the v7 trained foundation model.

Three modules:
- v7_inference: V7Foundation class — load model + maskable forward
- lookups: hero/item/account name lookups + user features
- queries: high-level query functions (personal_winprob, hero_pick_rec, etc.)

See notebook.qmd for an interactive demo.
"""

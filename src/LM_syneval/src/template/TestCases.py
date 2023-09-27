class TestCase:
    def __init__(self):
        self.agrmt_cases = [
            "obj_rel_across_anim",
            "obj_rel_across_inanim",
            "prep_anim",
            "prep_inanim",
            "simple_agrmt",
            "reflexives_across",
            "simple_reflexives",
            "reflexive_sent_comp",
        ]

        self.npi_cases = [
            "npi_across_anim",
            "npi_across_inanim",
            "simple_npi_anim",
            "simple_npi_inanim",
        ]

        self.all_cases = self.agrmt_cases + self.npi_cases

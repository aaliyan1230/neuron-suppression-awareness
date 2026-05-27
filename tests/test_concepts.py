from neuron_suppression_awareness.concepts import PROMPT_TEMPLATES, load_concepts


def test_concept_counts():
    split = load_concepts()
    assert len(split.train) == 150
    assert len(split.test) == 50


def test_no_overlap():
    split = load_concepts()
    overlap = set(split.train) & set(split.test)
    assert overlap == set(), f"Overlap: {overlap}"


def test_all_unique():
    split = load_concepts()
    all_concepts = list(split.train) + list(split.test)
    assert len(all_concepts) == len(set(all_concepts))


def test_categories_cover_all():
    split = load_concepts()
    from_categories = set()
    for words in split.categories.values():
        from_categories.update(words)
    all_concepts = set(split.train) | set(split.test)
    assert from_categories == all_concepts


def test_prompt_templates_have_placeholder():
    for tmpl in PROMPT_TEMPLATES:
        assert "{concept}" in tmpl, f"Missing placeholder in: {tmpl}"

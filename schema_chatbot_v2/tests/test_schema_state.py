from app.core.schema_state import SchemaState, normalize_type
from app.llm.base import FieldOp


def test_add_field_creates_skeleton_and_detects_gap():
    s = SchemaState()
    s.set_document_type("invoice")
    s.add_field("invoice_number")
    gap = s.next_gap()
    assert gap is not None
    assert gap.field_name == "invoice_number"
    assert gap.attribute == "type"


def test_field_becomes_complete_after_all_attrs_filled():
    s = SchemaState()
    s.add_field("total_amount")
    s.update_field_attribute("total_amount", "type", "number")
    assert s.next_gap().attribute == "required"
    s.update_field_attribute("total_amount", "required", True)
    assert s.next_gap() is None
    assert s.fields["total_amount"].is_complete()


def test_array_type_requires_item_type():
    s = SchemaState()
    s.add_field("line_items", type="array", required=True)
    gap = s.next_gap()
    assert gap.attribute == "item_type"
    s.update_field_attribute("line_items", "item_type", "string")
    assert s.next_gap() is None


def test_remove_field():
    s = SchemaState()
    s.add_field("gst")
    assert "gst" in s.fields
    assert s.remove_field("GST") is True  # normalization: case-insensitive
    assert "gst" not in s.fields
    assert s.remove_field("does_not_exist") is False


def test_field_order_preserved_and_gaps_asked_in_order():
    s = SchemaState()
    s.add_field("vendor_name")
    s.add_field("invoice_date")
    assert s.next_gap().field_name == "vendor_name"
    s.update_field_attribute("vendor_name", "type", "string")
    s.update_field_attribute("vendor_name", "required", True)
    assert s.next_gap().field_name == "invoice_date"


def test_add_field_merges_when_called_twice():
    s = SchemaState()
    s.add_field("amount", type="number")
    s.add_field("amount", required=True)
    assert s.fields["amount"].type == "number"
    assert s.fields["amount"].required is True
    assert len(s.field_order) == 1


def test_human_summary_contains_field_names():
    s = SchemaState()
    s.set_document_type("invoice")
    s.add_field("invoice_number", type="string", required=True)
    summary = s.human_summary()
    assert "invoice_number" in summary
    assert "invoice" in summary


# ---- all_gaps() - every open gap, not just the first ----

def test_all_gaps_returns_every_missing_attribute_across_fields():
    s = SchemaState()
    s.add_field("vendor_name")
    s.add_field("total_amount", type="number")  # still missing "required"
    gaps = s.all_gaps()
    assert [(g.field_name, g.attribute) for g in gaps] == [
        ("vendor_name", "type"),
        ("vendor_name", "required"),
        ("total_amount", "required"),
    ]


def test_all_gaps_empty_when_schema_complete():
    s = SchemaState()
    s.add_field("vendor_name", type="string", required=True)
    assert s.all_gaps() == []


# ---- normalize_type() ----

def test_normalize_type_passes_through_valid_type():
    assert normalize_type("number") == ("number", None)


def test_normalize_type_maps_known_alias():
    normalized, note = normalize_type("float")
    assert normalized == "number"
    assert "float" in note and "number" in note


def test_normalize_type_rejects_unknown_type_without_corrupting():
    normalized, note = normalize_type("frobnicate")
    assert normalized is None
    assert "frobnicate" in note


def test_normalize_type_none_passthrough():
    assert normalize_type(None) == (None, None)


# ---- apply_operations() - the "several things in one message" mechanism ----

def test_apply_operations_add_update_remove_in_one_batch():
    s = SchemaState()
    s.add_field("gst", type="string", required=True)
    s.add_field("total_amount")

    notes = s.apply_operations([
        FieldOp(op="add", field_name="vendor_name", type="string", required=True),
        FieldOp(op="update", field_name="total_amount", type="number", required=True),
        FieldOp(op="remove", field_name="gst"),
    ])

    assert notes == []
    assert "vendor_name" in s.fields and s.fields["vendor_name"].type == "string"
    assert s.fields["total_amount"].type == "number"
    assert s.fields["total_amount"].required is True
    assert "gst" not in s.fields


def test_apply_operations_normalizes_and_surfaces_a_note():
    s = SchemaState()
    s.add_field("total_amount")
    notes = s.apply_operations([FieldOp(op="update", field_name="total_amount", type="float")])
    assert s.fields["total_amount"].type == "number"
    assert any("float" in n for n in notes)


def test_apply_operations_update_on_missing_field_adds_it_instead():
    # Forgiving: the LLM may not always distinguish "update" vs "add"
    # correctly (e.g. answering a gap it thinks already exists).
    s = SchemaState()
    notes = s.apply_operations([FieldOp(op="update", field_name="new_field", type="string")])
    assert "new_field" in s.fields
    assert s.fields["new_field"].type == "string"

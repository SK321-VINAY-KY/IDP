from app.core.schema_state import SchemaState


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

import pytest

import simbus


def test_tops_param_filters_kind_and_dedupes():
    rows = [("sources", "/a"), ("obstacles", "/o"), ("sources", "/b"), ("sources", "/a"), ("forces", "")]
    assert simbus.tops_param(rows, "sources") == "/a /b"
    assert simbus.tops_param(rows, "obstacles") == "/o"
    assert simbus.tops_param(rows, "forces") == ""
    with pytest.raises(ValueError):
        simbus.tops_param(rows, "vel")


def test_grid_helpers():
    bounds = (-1.0, 0.0, 1.0, 2.0)
    assert simbus.matched_height(bounds, 512) == 512
    assert simbus.matched_height((0, 0, 2, 1), 512) == 256
    assert simbus.cell_size(bounds, (256, 256)) == (2 / 256, 2 / 256)
    uv = simbus.world_to_uv((0.0, 0.5), bounds)
    assert uv == (0.5, 0.25)
    assert simbus.uv_to_world(uv, bounds) == (0.0, 0.5)


def test_state_fields_have_formats():
    assert set(simbus.STATE_FIELDS) <= set(simbus.FIELDS)


class _Cell:
    def __init__(self, table, row, col):
        self._table = table
        self._row = row
        self._col = col

    @property
    def val(self):
        return self._table.rows[self._row][self._col]


class _Table:
    def __init__(self):
        self.rows = [["kind", "path"]]

    @property
    def numRows(self):
        return len(self.rows)

    def __getitem__(self, key):
        row, col = key
        return _Cell(self, row, col)

    def appendRow(self, row):
        self.rows.append(list(row))

    def deleteRow(self, row):
        del self.rows[row]


class _Merge:
    def __init__(self):
        self.par = type("Par", (), {"tops": ""})()


class _BusOwner:
    def __init__(self):
        self._ops = {"publishers": _Table()}
        for name in simbus.FieldBus.MERGE_TOPS.values():
            self._ops[name] = _Merge()

    def op(self, name):
        return self._ops[name]


class _Top:
    def __init__(self, path):
        self.path = path


def test_publish_and_unpublish_refresh_merges():
    owner = _BusOwner()
    bus = simbus.FieldBus(owner)
    with pytest.raises(ValueError):
        bus.Publish("vel", "/nope")
    assert owner.op("publishers").rows == [["kind", "path"]]

    bus.Publish("sources", "/a")
    bus.Publish("sources", _Top("/b"))
    bus.Publish("sources", "/a")
    bus.Publish("forces", "/f")
    assert owner.op("publishers").rows[1:] == [["sources", "/a"], ["sources", "/b"], ["forces", "/f"]]
    assert owner.op("sources_merge").par.tops == "/a /b"
    assert owner.op("forces_merge").par.tops == "/f"
    assert owner.op("obstacles_merge").par.tops == ""

    bus.Publish("obstacles", "/b")
    bus.Unpublish(_Top("/b"), "sources")
    assert owner.op("sources_merge").par.tops == "/a"
    assert owner.op("obstacles_merge").par.tops == "/b"
    bus.Unpublish("/b")
    assert owner.op("obstacles_merge").par.tops == ""
    assert all(row[1] != "/b" for row in owner.op("publishers").rows)

"""Canonical metadata views shared by controls, plots and exported tables.

Raw metadata is never mutated. Conflicting aliases are errors, not a preference
for whichever spelling happens to be encountered first.
"""
from decimal import Decimal, InvalidOperation

GROUP_FIELDS = ("cell_id", "sample_id", "direction", "cycle", "temperature_C",
                "protocol_id", "ac_amplitude_mv", "rest_time_s", "pressure_mpa")
ALIASES = {
    "temperature_C": ("temperature_c",),
    "protocol_id": ("protocol_id", "protocol"),
    "ac_amplitude_mv": ("ac_amplitude_mv", "amplitude_mv", "perturbation_mv"),
    "rest_time_s": ("rest_time_s", "rest_before_eis_s", "ocv_rest_s", "hold_time_s"),
    "voltage_v": ("voltage_v", "potential_v"),
}
NUMERIC = {"cycle", "temperature_C", "ac_amplitude_mv", "rest_time_s",
           "pressure_mpa", "voltage_v", "target_voltage_v", "soc_percent", "area_cm2"}
REVERSE = {alias: key for key, aliases in ALIASES.items() for alias in aliases}


def canonical_metadata(metadata):
    result = {}
    for raw_key, raw_value in metadata.items():
        if raw_value is None or not str(raw_value).strip():
            continue
        key = str(raw_key).strip().lower()
        key = REVERSE.get(key, key)
        value = str(raw_value).strip()
        if key in NUMERIC:
            try:
                number = Decimal(value)
                if not number.is_finite():
                    raise ValueError("Nonfinite metadata: " + key)
                value = format(number.normalize(), "f") if number else "0"
            except InvalidOperation as exc:
                if key == "cycle":
                    # Some instruments use a cycle label instead of an integer.
                    result[key] = value
                    continue
                raise ValueError("Nonnumeric metadata: " + key) from exc
        if key in result and result[key] != value:
            raise ValueError("Conflicting metadata aliases: " + key)
        result[key] = value
    if "voltage_v" not in result and "target_voltage_v" in result:
        result["voltage_v"] = result["target_voltage_v"]
    return result


def condition_key(result):
    m = canonical_metadata(result.get("metadata", {}))
    # Plot eligibility requires an explicit identity. Tables can retain anonymous
    # records, but never merge them across source identities.
    if not (m.get("cell_id") or m.get("sample_id")):
        m["cell_id"] = "source:" + str(result.get("source_uid") or result.get("spectrum_uid"))
    return tuple(m.get(k, "") for k in GROUP_FIELDS) + (
        result.get("impedance_basis"), result.get("display_role"),
        m.get("data_origin", "experimental-or-unspecified"))

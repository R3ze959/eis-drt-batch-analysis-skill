#!/usr/bin/env python3
"""PUB-01/PUB-07 regression tests; synthetic data, no private or holdout inputs."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eis_parser import TextParseError, detect_text_table, file_hash, qc_for, read_text
from result_contract import export_bundle, identities, normalized_input_rows, numerical_digest


def options(**changes):
    values = dict(frequency_unit="hz", impedance_unit="ohm", imag_convention="auto",
                  text_columns=None, text_headerless=False, text_delimiter="auto", text_header_row=None)
    values.update(changes)
    return argparse.Namespace(**values)


class TextParserContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="drt-parser-contract-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "observations.csv"

    def parse(self, content, **changes):
        self.path.write_text(content, encoding="utf-8")
        before = file_hash(self.path)
        spectra, audit = read_text(self.path, "same-label", options(**changes))
        self.assertEqual(file_hash(self.path), before, "Parsing must not mutate the raw input")
        return spectra, audit

    def headerless_options(self, **changes):
        values = dict(text_columns={"frequency": 0, "zreal": 1, "zimag": 2},
                      text_headerless=True, text_delimiter="comma", imag_convention="zimag")
        values.update(changes)
        return values

    def test_z_prime_aliases_preserve_mixed_imaginary_sign(self):
        spectra, audit = self.parse("freq;Z_prime;Z_double_prime\n1000;1;0.2\n10;2;-0.4\n1;3;-0.1\n")
        spec = spectra[0]
        np.testing.assert_array_equal(spec.freq_hz, [1000, 10, 1])
        np.testing.assert_array_equal(spec.zreal, [1, 2, 3])
        np.testing.assert_array_equal(spec.neg_zimag, [-0.2, 0.4, 0.1])
        np.testing.assert_array_equal(spec.acquisition_index, [2, 3, 4])
        self.assertEqual(audit["column_mapping"], {"frequency": 0, "zreal": 1, "zimag": 2})
        self.assertTrue(qc_for(spec)["row_accounting_matches_raw"])

    def test_indexed_freq_real_imag_keeps_first_observation(self):
        spectra, _ = self.parse(",Freq,Real,Imag\n0,0.01,109.2,0.58\n1,0.1,109.6,-1.11\n2,1,100,-2\n")
        spec = spectra[0]
        np.testing.assert_array_equal(spec.freq_hz, [0.01, 0.1, 1])
        np.testing.assert_array_equal(spec.neg_zimag, [-0.58, 1.11, 2])
        self.assertEqual(spec.raw_rows, 3)

    def test_negative_imaginary_aliases_have_no_double_sign_flip(self):
        for header in ("-Z_double_prime", "-Imag", "negImag", "minusImag", "negImZ", "-Z″"):
            with self.subTest(header=header):
                spectra, _ = self.parse(f"Freq,Real,{header}\n10,2,0.3\n1,4,-0.5\n")
                np.testing.assert_array_equal(spectra[0].neg_zimag, [0.3, -0.5])

    def test_explicit_units_convert_arrays_and_unicode_prime_header(self):
        spectra, _ = self.parse("Freq(kHz),Z′(mohm),Z″(mohm)\n1,1000,-200\n0.1,2000,100\n",
                                frequency_unit="auto", impedance_unit="auto")
        np.testing.assert_array_equal(spectra[0].freq_hz, [1000, 100])
        np.testing.assert_array_equal(spectra[0].zreal, [1, 2])
        np.testing.assert_allclose(spectra[0].neg_zimag, [0.2, -0.1])

    def test_aliases_do_not_supply_unstated_units(self):
        for field in ("frequency_unit", "impedance_unit"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unit is ambiguous"):
                self.parse("Freq,Real,Imag\n10,2,-0.3\n", **{field: "auto"})

    def test_headerless_preserves_first_row_and_original_line_order(self):
        spectra, audit = self.parse("1000,1,0.2\n10,2,-0.4\n1,3,-0.1\n", **self.headerless_options())
        spec = spectra[0]
        np.testing.assert_array_equal(spec.freq_hz, [1000, 10, 1])
        np.testing.assert_array_equal(spec.zreal, [1, 2, 3])
        np.testing.assert_array_equal(spec.neg_zimag, [-0.2, 0.4, 0.1])
        np.testing.assert_array_equal(spec.acquisition_index, [1, 2, 3])
        self.assertEqual(spec.raw_rows, 3)
        self.assertIsNone(audit["header_line"])
        self.assertEqual(audit["first_data_line"], 1)
        self.assertTrue(audit["headerless"])

    def test_headerless_accounting_retains_invalid_rows_as_exclusions(self):
        content = "1000,1,-1\n\n100,2,NaN\n0,3,-3\n10\n1,4,-4\n"
        spectra, _ = self.parse(content, **self.headerless_options())
        spec = spectra[0]
        np.testing.assert_array_equal(spec.freq_hz, [1000, 1])
        np.testing.assert_array_equal(spec.acquisition_index, [1, 6])
        self.assertEqual(spec.raw_rows, 6)
        self.assertEqual(spec.excluded, {"blank_row": 1, "nonnumeric_or_nonfinite": 1,
                                         "nonpositive_frequency": 1, "short_row": 1})
        self.assertTrue(qc_for(spec)["row_accounting_matches_raw"])

    def test_headerless_requires_explicit_map_delimiter_units_and_sign(self):
        cases = ({"text_columns": None}, {"text_delimiter": "auto"},
                 {"frequency_unit": "auto"}, {"impedance_unit": "auto"},
                 {"imag_convention": "auto"}, {"text_header_row": 1},
                 {"text_columns": {"frequency": "Freq", "zreal": 1, "zimag": 2}})
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.parse("10,1,-1\n1,2,-2\n", **self.headerless_options(**change))

    def test_numeric_first_row_is_never_implicitly_a_header(self):
        content = "1000,1,-1\n100,2,-2\n"
        with self.assertRaisesRegex(ValueError, "headerless"):
            self.parse(content)
        with self.assertRaisesRegex(ValueError, "retain the first observation"):
            self.parse(content, text_columns={"frequency": 0, "zreal": 1, "zimag": 2},
                       text_delimiter="comma", imag_convention="zimag")

    def test_nonfinite_numeric_first_row_is_not_a_header_either(self):
        for value in ("NaN", "Inf", "-Inf"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "retain the first observation"):
                self.parse(f"1000,1,{value}\n100,2,-2\n",
                           text_columns={"frequency": 0, "zreal": 1, "zimag": 2},
                           text_delimiter="comma", imag_convention="zimag")

    def test_headerless_whitespace_and_explicit_negative_sign(self):
        spectra, _ = self.parse("1000 1 0.2\n10\t2\t-0.4\n",
                                **self.headerless_options(text_delimiter="whitespace", imag_convention="neg-zimag"))
        np.testing.assert_array_equal(spectra[0].freq_hz, [1000, 10])
        np.testing.assert_array_equal(spectra[0].neg_zimag, [0.2, -0.4])

    def test_explicit_map_uses_exact_header_names_and_keeps_acquisition_index(self):
        mapping = {"frequency": "drive", "zreal": "component A", "zimag": "component B", "index": "id"}
        spectra, audit = self.parse("id;component B;drive;component A\n7;-200;1;1000\n11;100;0.1;2000\n",
                                   text_columns=json.dumps(mapping), text_delimiter="semicolon",
                                   imag_convention="zimag", frequency_unit="khz", impedance_unit="mohm")
        spec = spectra[0]
        np.testing.assert_array_equal(spec.freq_hz, [1000, 100])
        np.testing.assert_array_equal(spec.zreal, [1, 2])
        np.testing.assert_allclose(spec.neg_zimag, [0.2, -0.1])
        np.testing.assert_array_equal(spec.acquisition_index, [7, 11])
        self.assertEqual(audit["column_mapping"], {"frequency": 2, "zreal": 3, "zimag": 1, "index": 0})

    def test_unknown_component_header_needs_declared_sign(self):
        with self.assertRaisesRegex(ValueError, "Imaginary sign is ambiguous"):
            self.parse("a,b,c\n100,1,-2\n", text_columns={"frequency": 0, "zreal": 1, "zimag": 2},
                       text_delimiter="comma")

    def test_declared_header_row_preserves_physical_source_line_numbers(self):
        content = "instrument note\noperator note\na;b;c\n10;2;-1\n1;3;-2\n"
        spectra, audit = self.parse(content, text_columns={"frequency": 0, "zreal": 1, "zimag": 2},
                                   text_delimiter="semicolon", text_header_row=3, imag_convention="zimag")
        np.testing.assert_array_equal(spectra[0].acquisition_index, [4, 5])
        np.testing.assert_array_equal(spectra[0].freq_hz, [10, 1])
        self.assertEqual(audit["header_line"], 3)

    def test_ambiguous_duplicate_components_require_a_map(self):
        content = "Freq,Real,Imag,Imag\n10,1,-2,-200\n1,3,-4,-400\n"
        with self.assertRaisesRegex(ValueError, "Ambiguous duplicate"):
            self.parse(content)
        spectra, _ = self.parse(content, text_columns={"frequency": 0, "zreal": 1, "zimag": 2},
                                text_delimiter="comma")
        np.testing.assert_array_equal(spectra[0].neg_zimag, [2, 4])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.parse(content, text_columns={"frequency": "Freq", "zreal": "Real", "zimag": "Imag"},
                       text_delimiter="comma")

    def test_invalid_maps_and_header_rows_are_rejected(self):
        maps = [{"frequency": 0, "zreal": 0, "zimag": 2},
                {"frequency": 0, "zreal": 1, "zimag": 4},
                {"frequency": 0, "zreal": 1, "zimag": -1},
                {"frequency": 0, "zreal": True, "zimag": 2},
                {"frequency": 0, "zreal": 1, "zimag": 2, "unused": 3},
                '{"frequency":0,"frequency":1,"zreal":1,"zimag":2}',
                {"frequency": "freq", "zreal": "Real", "zimag": "Imag"}]
        for mapping in maps:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.parse("Freq,Real,Imag\n10,1,-2\n", text_columns=mapping, text_delimiter="comma")
        for row in (0, -1, 3, True, "1"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                detect_text_table(["Freq,Real,Imag", "10,1,-2"], header_row=row)

    def test_frequency_restarts_remain_separate_spectra(self):
        spectra, _ = self.parse("100,1,-1\n10,2,-2\n1,3,-3\n100,4,-4\n10,5,-5\n1,6,-6\n",
                                **self.headerless_options())
        self.assertEqual(len(spectra), 2)
        np.testing.assert_array_equal(spectra[0].zreal, [1, 2, 3])
        np.testing.assert_array_equal(spectra[1].zreal, [4, 5, 6])
        np.testing.assert_array_equal(spectra[1].acquisition_index, [4, 5, 6])

    def test_complete_physical_line_ledger_retains_each_exclusion(self):
        content = '=instrument-note\nFreq,Real,Imag\n1000,1,-1\n\n100,2,NaN\n0,3,-3\n10\n"unterminated\n1,4,-4\n'
        spectra, audit = self.parse(content)
        ledger = audit["row_ledger"]
        self.assertEqual([r["source_line"] for r in ledger], list(range(1, 10)))
        self.assertEqual([r["raw_line"] for r in ledger], content.splitlines())
        self.assertEqual([r["status"] for r in ledger[:2]], ["preamble", "header"])
        self.assertEqual([(r["source_line"], r["reason"]) for r in ledger if r["status"] == "excluded"],
                         [(4, "blank_row"), (5, "nonnumeric_or_nonfinite"), (6, "nonpositive_frequency"),
                          (7, "short_row"), (8, "csv_parse_error")])
        self.assertEqual(ledger[4]["raw_values"]["zimag"], "NaN")
        self.assertTrue(audit["row_accounting"]["all_source_lines_accounted"])
        self.assertTrue(audit["row_accounting"]["table_rows_accounted"])
        self.assertTrue(qc_for(spectra[0])["row_accounting_matches_raw"])
        json.dumps(audit, allow_nan=False)

    def test_all_invalid_source_failure_still_carries_row_audit(self):
        content = 'Freq,Real,Imag\n0,1,-2\n3,NaN,-1\n\n'
        with self.assertRaisesRegex(TextParseError, "No valid EIS points") as caught:
            self.parse(content)
        audit = caught.exception.audit
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["row_accounting"]["excluded_rows"], 3)
        self.assertEqual(audit["parsed_spectra"], 0)
        self.assertEqual([r["raw_line"] for r in audit["row_ledger"]], content.splitlines())
        self.assertEqual(file_hash(self.path), audit["sha256"])
        json.dumps(audit, allow_nan=False)

    def test_header_detection_failure_preserves_all_unparsed_lines(self):
        content = 'equipment note\n1,2,3\n4,5,6\n'
        with self.assertRaises(TextParseError) as caught:
            self.parse(content)
        ledger = caught.exception.audit["row_ledger"]
        self.assertEqual([r["source_line"] for r in ledger], [1, 2, 3])
        self.assertTrue(all(r["status"] == "unparsed" and r["reason"] for r in ledger))

    def test_ambiguous_sign_records_valid_candidates_as_undelivered(self):
        with self.assertRaisesRegex(TextParseError, "Imaginary sign is ambiguous") as caught:
            self.parse('a,b,c\n10,1,-2\n1,2,-3\n',
                       text_columns={"frequency": 0, "zreal": 1, "zimag": 2}, text_delimiter="comma")
        audit = caught.exception.audit
        self.assertEqual([r["status"] for r in audit["row_ledger"]],
                         ["header", "parsed-not-delivered", "parsed-not-delivered"])
        self.assertEqual(audit["row_accounting"]["included_rows"], 0)
        self.assertEqual(audit["row_accounting"]["undelivered_rows"], 2)
        self.assertTrue(audit["row_accounting"]["table_rows_accounted"])

    def test_invalid_declared_valid_flags_are_not_silently_accepted(self):
        spectra, audit = self.parse('Freq,Real,Imag,Valid\n100,1,-1,1\n10,2,-2,0\n5,2,-2,nope\n3,2,-2,NaN\n2,2,-2\n1,3,-3,1\n')
        np.testing.assert_array_equal(spectra[0].freq_hz, [100, 1])
        self.assertEqual(audit["excluded_by_reason"], {"Valid=0": 1, "invalid_valid_flag": 3})
        self.assertEqual([r["source_line"] for r in audit["row_ledger"] if r["status"] == "excluded"], [3, 4, 5, 6])

    def test_finite_input_that_overflows_unit_conversion_is_excluded(self):
        spectra, audit = self.parse('Freq,Real,Imag\n1e308,1,-1\n1,1e308,-1\n.1,2,-3\n',
                                   frequency_unit="khz", impedance_unit="kohm")
        self.assertEqual(audit["excluded_by_reason"], {"nonfinite_after_unit_conversion": 2})
        np.testing.assert_array_equal(spectra[0].freq_hz, [100])
        np.testing.assert_array_equal(spectra[0].zreal, [2000])
        json.dumps(audit, allow_nan=False)

    def test_raw_strings_are_retained_separately_from_normalized_units(self):
        spectra, audit = self.parse('Freq(kHz),Z′(mohm),Z″(mohm),note\n1.00,1000,-200,"=1+2, raw"\n.10,2000,100,other\n',
                                   frequency_unit="auto", impedance_unit="auto")
        row = audit["row_ledger"][1]
        self.assertEqual(row["raw_fields"], ["1.00", "1000", "-200", "=1+2, raw"])
        self.assertEqual(row["normalized_values"]["frequency_hz"], 1000)
        self.assertEqual(row["normalized_values"]["zreal"], 1)
        self.assertAlmostEqual(row["normalized_values"]["neg_zimag"], .2)
        self.assertEqual(spectra[0].metadata["included_raw_values"][0]["frequency"], "1.00")

    def test_multirun_exclusions_are_source_scoped_without_guessing_owner(self):
        spectra, audit = self.parse('Freq,Real,Imag\n100,1,-1\n10,2,-2\n1,3,-3\ninvalid\n100,4,-4\n10,5,-5\n1,6,-6\n\n')
        self.assertEqual(len(spectra), 2)
        self.assertEqual(audit["row_accounting"]["included_rows"], 6)
        self.assertEqual(audit["row_accounting"]["excluded_rows"], 2)
        self.assertTrue(audit["row_accounting"]["table_rows_accounted"])
        self.assertTrue(all("spectrum_id" not in r for r in audit["row_ledger"] if r["status"] == "excluded"))
        self.assertEqual(spectra[1].metadata["included_source_lines"], [6, 7, 8])

    def test_explicit_single_spectrum_sort_preserves_values_and_line_mapping(self):
        content = 'Freq,Real,Imag,Index\n10,2,-2,42\n1000,1,-1,31\n1,4,-4,77\n100,3,-3,50\n.1,5,-5,100\n'
        spectra, audit = self.parse(content, frequency_order="single-spectrum-descending")
        spec = spectra[0]
        self.assertEqual(len(spectra), 1)
        np.testing.assert_array_equal(spec.freq_hz, [1000, 100, 10, 1, .1])
        np.testing.assert_array_equal(spec.zreal, [1, 3, 2, 4, 5])
        np.testing.assert_array_equal(spec.acquisition_index, [31, 50, 42, 77, 100])
        self.assertEqual(spec.metadata["included_source_lines"], [3, 5, 2, 4, 6])
        self.assertEqual([r["zreal"] for r in spec.metadata["included_raw_values"]], ["1", "3", "2", "4", "5"])
        self.assertEqual([r["source_line"] for r in audit["row_ledger"]], list(range(1, 7)))
        self.assertTrue(audit["sorting_applied"])
        self.assertFalse(spec.metadata["analysis_order_matches_acquisition"])

    def test_default_unordered_input_keeps_legacy_restart_partition_and_warning(self):
        spectra, audit = self.parse('Freq,Real,Imag\n10,2,-2\n1000,1,-1\n1,4,-4\n100,3,-3\n.1,5,-5\n')
        self.assertGreater(len(spectra), 1)
        np.testing.assert_array_equal(np.concatenate([s.freq_hz for s in spectra]), [10, 1000, 1, 100, .1])
        self.assertTrue(audit["frequency_order_warnings"])
        self.assertFalse(audit["sorting_applied"])

    def test_explicit_sort_refuses_duplicate_and_repeated_sweeps(self):
        for content in ('100,1,-1\n100,2,-2\n1,3,-3\n',
                        '100,1,-1\n10,2,-2\n1,3,-3\n100,4,-4\n10,5,-5\n1,6,-6\n'):
            with self.subTest(content=content), self.assertRaisesRegex(TextParseError, "refuses duplicate") as caught:
                self.parse(content, **self.headerless_options(frequency_order="single-spectrum-descending"))
            self.assertEqual(caught.exception.audit["row_accounting"]["included_rows"], 0)
            self.assertEqual(len(caught.exception.audit["row_ledger"]), len(content.splitlines()))

    def test_sorted_time_diagnostics_still_follow_physical_acquisition_order(self):
        spectra, _ = self.parse('Freq,Real,Imag,Time(s)\n10,2,-2,1\n100,1,-1,2\n1,3,-3,3\n',
                               frequency_order="single-spectrum-descending")
        metadata = spectra[0].metadata
        self.assertEqual(metadata["acquisition_time_values_s"], [2, 1, 3])
        self.assertTrue(metadata["time_monotonic_increasing"])
        self.assertEqual(metadata["sweep_duration_s"], 2)
        self.assertEqual(metadata["time_step_anomaly_count"], 0)

    def test_time_anomaly_source_lines_survive_sorting_and_missing_optional_time(self):
        spectra, _ = self.parse('Freq,Real,Imag,Time(s)\n10,2,-2,0\n100,1,-1,NaN\n1,3,-3,1\n.1,4,-4,100\n.01,5,-5,101\n',
                               frequency_order="single-spectrum-descending")
        metadata = spectra[0].metadata
        self.assertEqual(metadata["acquisition_time_diagnostics_source_lines"], [2, 4, 5, 6])
        anomaly = metadata["time_step_anomalies"][0]
        self.assertEqual((anomaly["from_source_line"], anomaly["to_source_line"]), (4, 5))

    def test_missing_optional_phase_is_audited_without_losing_valid_eis(self):
        spectra, audit = self.parse('Freq,Real,Imag,Phase\n100,1,-1,NaN\n10,2,-2,bad\n1,3,-3,\n')
        np.testing.assert_array_equal(spectra[0].neg_zimag, [1, 2, 3])
        self.assertEqual(audit["row_accounting"]["included_rows"], 3)
        self.assertEqual(audit["row_ledger"][2]["raw_values"]["phase"], 'bad')

    def test_invalid_order_policy_fails_with_audit(self):
        with self.assertRaisesRegex(TextParseError, "frequency_order must") as caught:
            self.parse('Freq,Real,Imag\n10,1,-1\n', frequency_order="guess")
        self.assertEqual(caught.exception.audit["row_accounting"]["undelivered_rows"], 1)

    def test_decode_failure_is_file_failure_not_claimed_complete_empty_ledger(self):
        self.path.write_bytes(b'Freq,Real,Imag\n10,1,\xff\n')
        with patch('eis_parser.detect_encoding', return_value='utf-8'), self.assertRaises(TextParseError) as caught:
            read_text(self.path, 'bad-encoding', options())
        audit = caught.exception.audit
        self.assertFalse(audit["source_line_ledger_available"])
        self.assertFalse(audit["row_accounting"]["all_source_lines_accounted"])
        self.assertEqual(audit["sha256"], file_hash(self.path))
        self.assertIn("UnicodeDecodeError", audit["error"])


def export_pair(source, offset=0, failed=False):
    provenance = {**identities(source, "content-hash", "same-internal-id"),
                  "source": str(source), "source_sha256": "content-hash", "spectrum_id": "same-internal-id",
                  "label": "same-label", "impedance_basis": "ohm", "metadata": {"data_origin": "synthetic"}}
    payload = {**provenance, "acquisition_index": [1, 4], "frequency_hz": [100., 1.],
               "zreal": [1. + offset, 2. + offset], "neg_zimag": [-0.1, 0.2]}
    result = {**provenance, "status": "failed" if failed else "completed"}
    if failed:
        result["error"] = "synthetic inversion failure"
    else:
        result.update({"display_role": "diagnostic-only", "validity_status": "controls-incomplete-no-acceptance",
                       "numeric_status": "numeric-supported", "evidence_status": "controls-incomplete",
                       "recommended_branch": "signed_gdrt", "recommendation_status": "provisional",
                       "recommended_rms_percent": 0.25, "drt": {"lambda": 1e-4},
                       "curves": {"frequency_hz": [100., 1.], "tau_s": [0.01, 0.1], "gamma": [3., 4.]}})
    return payload, result


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class NormalizedExportContractTests(unittest.TestCase):
    def test_uid_join_not_labels_source_ids_or_result_order(self):
        first, a = export_pair("/synthetic/a/input.csv")
        second, b = export_pair("/synthetic/b/input.csv", offset=10)
        b.update(display_role="included", validity_status="accepted-with-supplied-controls")
        rows = normalized_input_rows([first, second], [b, a])
        self.assertNotEqual(first["spectrum_uid"], second["spectrum_uid"])
        self.assertEqual([row["display_role"] for row in rows], ["diagnostic-only"] * 2 + ["included"] * 2)
        np.testing.assert_array_equal([row["zreal"] for row in rows], [1, 2, 11, 12])
        np.testing.assert_array_equal([row["acquisition_index"] for row in rows], [1, 4, 1, 4])

    def test_exported_normalized_rows_share_index_verdict_and_keep_primary_meaning(self):
        first, a = export_pair("/synthetic/a/input.csv")
        second, b = export_pair("/synthetic/b/input.csv", offset=10, failed=True)
        original_payloads = copy.deepcopy([first, second])
        with tempfile.TemporaryDirectory(prefix="drt-export-contract-") as directory:
            out = Path(directory) / "output"
            out.mkdir()
            (out / "execution_contract.json").write_text("{}", encoding="utf-8")
            export_bundle(out, [b, a], [first, second], [], {}, {})
            normalized = csv_rows(out / "01_inputs/normalized_eis.csv")
            index = {row["spectrum_uid"]: row for row in csv_rows(out / "00_overview/spectrum_index.csv")}
            fields = ("status", "display_role", "validity_status", "numeric_status", "evidence_status",
                      "recommended_branch", "recommendation_status", "recommended_rms_percent")
            for row in normalized:
                for field in fields:
                    self.assertEqual(row[field], index[row["spectrum_uid"]][field])
                self.assertEqual(row["verdict_join"], "spectrum_uid")
            failed_rows = [row for row in normalized if row["spectrum_uid"] == b["spectrum_uid"]]
            self.assertEqual(len(failed_rows), 2)
            self.assertTrue(all(row["numeric_status"] == "failed" and row["error"] for row in failed_rows))
            np.testing.assert_array_equal([float(row["zreal"]) for row in normalized], [1, 2, 11, 12])
            primary = csv_rows(out / "03_results/primary_drt/curves.csv")
            np.testing.assert_array_equal([float(row["gamma"]) for row in primary], [3, 4])
            self.assertEqual({row["spectrum_uid"] for row in primary}, {a["spectrum_uid"]})
        self.assertEqual([first, second], original_payloads)

    def test_failed_results_have_explicit_nonacceptance_states(self):
        payload, failed = export_pair("/synthetic/failure.csv", failed=True)
        rows = normalized_input_rows([payload], [failed])
        for row in rows:
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["numeric_status"], "failed")
            self.assertEqual(row["validity_status"], "review-required")
            self.assertEqual(row["evidence_status"], "not-assessed")
            self.assertEqual(row["display_role"], "review-pending")

    def test_missing_and_duplicate_uids_are_rejected(self):
        payload, result = export_pair("/synthetic/input.csv")
        cases = [([payload], []), ([payload, payload], [result]), ([payload], [result, result]),
                 ([{**payload, "spectrum_uid": None}], [result]),
                 ([payload], [{**result, "spectrum_uid": None}])]
        for payloads, results in cases:
            with self.subTest(payloads=len(payloads), results=len(results)), self.assertRaises(ValueError):
                normalized_input_rows(payloads, results)

    def test_uid_match_does_not_suppress_provenance_conflicts(self):
        payload, result = export_pair("/synthetic/input.csv")
        for field in ("source_uid", "source_sha256", "spectrum_id", "source", "impedance_basis"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "conflicts"):
                normalized_input_rows([payload], [{**result, field: "different"}])

    def test_array_misalignment_is_rejected_not_silently_truncated(self):
        payload, result = export_pair("/synthetic/input.csv")
        for values in ([1], [[1, 2]], 1):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "not aligned"):
                normalized_input_rows([{**payload, "zreal": values}], [result])

    def test_new_numerical_evidence_changes_digest(self):
        _, result = export_pair("/synthetic/input.csv")
        baseline = numerical_digest(result)
        for field in ("peak_evidence", "model_recommendation", "loewner_validation"):
            with self.subTest(field=field):
                self.assertNotEqual(numerical_digest({**result, field: {"status": "checked"}}), baseline)


if __name__ == "__main__":
    unittest.main(verbosity=2)

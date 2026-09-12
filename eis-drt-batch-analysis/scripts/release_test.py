#!/usr/bin/env python3
"""Public-release tests: independent parser, privacy, profiles, and visual contracts."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from batch_drt import DEFAULT_PARSER, apply_reference_profile
from eis_parser import read_text, read_irf, qc_for
from plot_drt_trends import prepare_groups, clip_group, cell_edges, render_trends, axis_qa, plt
from result_contract import numerical_digest
from generate_demo import generate
from build_release import build, collect, privacy_findings, verify_archive

ROOT = Path(__file__).resolve().parents[1]


def fixture(uid="s0", voltage=3.1, **changes):
    t = np.logspace(-6, 1, 160)
    g = 3*np.exp(-((np.log10(t)+3)**2)/.10) + 8*np.exp(-((np.log10(t)+1.5)**2)/.25)
    r = {"status": "completed", "spectrum_uid": uid, "ordinary_rc_eligible": True,
         "display_role": "included-exploratory", "impedance_basis": "ohm",
         "metadata": {"cell_id": "TEST", "direction": "charge", "voltage_v": voltage,
                      "data_origin": "synthetic"},
         "curves": {"tau_s": t.tolist(), "gamma": g.tolist(),
                    "frequency_hz": np.logspace(5,-1,50).tolist()}}
    r.update(changes)
    return r


class ParserTests(unittest.TestCase):
    def test_parser_is_bundled(self):
        self.assertEqual(DEFAULT_PARSER.resolve(), ROOT/"scripts/eis_parser.py")
        self.assertTrue(DEFAULT_PARSER.is_file())

    def test_text_units_and_acquisition_order(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"test.tsv"
            f.write_text("Frequency (kHz)\tZreal (mohm)\tZimag (mohm)\n1\t1000\t-100\n.1\t2000\t-200\n")
            a=argparse.Namespace(imag_convention="auto",frequency_unit="auto",impedance_unit="auto")
            s,_=read_text(f,"fixture",a)
            np.testing.assert_allclose(s[0].freq_hz,[1000,100])
            np.testing.assert_allclose(s[0].zreal,[1,2])
            np.testing.assert_allclose(s[0].neg_zimag,[.1,.2])
            self.assertTrue(qc_for(s[0])["row_accounting_matches_raw"])

    def test_irf_database_without_external_skill(self):
        with tempfile.TemporaryDirectory() as d:
            db=Path(d)/"data.db"
            con=sqlite3.connect(db)
            con.execute('CREATE TABLE EIS ("Index" INTEGER, VerifiedAppliedFrequency REAL, ImpedanceReal REAL, ImpedanceImaginary REAL)')
            con.executemany("INSERT INTO EIS VALUES (?, ?, ?, ?)", [(1,1000,1,-.1),(2,100,2,-.2)])
            con.commit();con.close()
            archive=Path(d)/"test.irf"
            with zipfile.ZipFile(archive,"w") as z:z.write(db,"fixture.db")
            spectra,audit=read_irf(archive,"synthetic","zimag")
            np.testing.assert_allclose(spectra[0].neg_zimag,[.1,.2])
            self.assertEqual(len(spectra),1)
            self.assertTrue(audit["sha256"])

    def test_demo_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as d:
            generate(Path(d)/"demo",2)
            with self.assertRaises(FileExistsError):generate(Path(d)/"demo",2)

    def test_external_profile_is_explicit_and_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"profile.json"
            p.write_text(json.dumps({"name":"test-external","lambda_policy":"fixed","fixed_lambda":1e-4,
              "derivative_order":1,"rbf_type":"gaussian","rbf_shape":"fwhm","shape_coeff":.5,
              "inductance_policy":"adaptive"}))
            args=argparse.Namespace(profile_json=p,reference_profile="none")
            apply_reference_profile(args)
            self.assertEqual(args.reference_profile,"test-external")
            self.assertEqual(len(args.profile_sha256),64)
            self.assertEqual(args.inductance_policy,"adaptive")

    def test_external_profile_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"profile.json";p.write_text('{"run_shell": "forbidden"}')
            with self.assertRaises(ValueError):
                apply_reference_profile(argparse.Namespace(profile_json=p,reference_profile="none"))


class PackagingTests(unittest.TestCase):
    def test_home_mount_and_encoded_path_detection(self):
        from urllib.parse import quote_from_bytes
        cases = [b"/" + b"Users/example/input.csv", b"/" + b"home/example/input.csv",
                 b"/" + b"Volumes/lab/input.csv", b"/" + b"mnt/lab/input.csv",
                 b"C:" + bytes([92]) + b"Users" + bytes([92]) + b"example",
                 b"/" + b"private/" + b"var/" + b"folders/example/image.png"]
        for value in cases:
            for encoded in (value, quote_from_bytes(value, safe="").encode(),
                            quote_from_bytes(quote_from_bytes(value, safe="").encode(), safe="").encode(),
                            value.replace(b"/", bytes([92]) + b"u002f"),
                            value.replace(bytes([92]), bytes([92, 92]))):
                with self.subTest(value=encoded):
                    self.assertIn("private-case-or-machine-path", privacy_findings(encoded))

    def test_generic_secret_detection(self):
        cases = [(b"gh" + b"p_" + b"x"*24, "possible-secret"),
                 (b"github_" + b"pat_" + b"x"*24, "possible-secret"),
                 (b"AK" + b"IA" + b"A"*16, "possible-secret"),
                 (b"-----BEGIN RSA PRIVATE " + b"KEY-----", "possible-secret"),
                 (b"https://" + b"example:password@" + b"example.invalid", "credential-bearing-url")]
        for value, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, privacy_findings(value))

    def test_generic_placeholders_and_upstream_attribution_are_allowed(self):
        for value in (b"/absolute/input", b"C:" + bytes([92]) + b"data",
                      b"Author <author@example.org>", b"https://example.org/method"):
            self.assertEqual(privacy_findings(value), [])

    def test_private_and_home_paths_rejected(self):
        for data in [b"/" + b"Users/example/file", b"/" + b"home/example/data",
                     b"/" + b"Volumes/example/data", b"wx" + b"id_123"]:
            self.assertTrue(privacy_findings(data))

    def test_allowlist_zip_is_repeatable_and_verified(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"source";root.mkdir()
            (root/"VERSION").write_text("0.1.0-beta.1\n")
            (root/"SKILL.md").write_text("Generic synthetic test fixture\n")
            (root/"release_manifest.json").write_text(json.dumps({
                "files": ["VERSION","SKILL.md","release_manifest.json"]}))
            (root/"not-listed-private.txt").write_text("do not bundle this file")
            a=build(Path(d)/"a",root);b=build(Path(d)/"b",root)
            self.assertEqual(a.read_bytes(),b.read_bytes())
            self.assertEqual(verify_archive(a, root)["status"],"PASS")
            with zipfile.ZipFile(a) as z:
                self.assertFalse(any("not-listed" in n for n in z.namelist()))
            (root/"SKILL.md").write_bytes(b"/" + b"Users/example/private-data")
            with self.assertRaises(ValueError):collect(root)

    def test_source_release_members_are_clean(self):
        self.assertGreater(len(collect(ROOT)),30)


class PlotContractTests(unittest.TestCase):
    def test_right_side_voltage_labels_are_in_qa(self):
        fig,ax=plt.subplots(figsize=(6,4))
        ax.set_ylim(0,1)
        ax.set_yticks([.5,.501],["first-right","second-right"])
        ax.yaxis.tick_right()
        qa=axis_qa(fig,ax)
        plt.close(fig)
        self.assertIn(["first-right","second-right"],qa["text_overlaps"])

    def test_identity_required(self):
        r=fixture();r["metadata"].pop("cell_id")
        groups, mapping=prepare_groups([(Path("r.json"),r)])
        self.assertFalse(groups)
        self.assertEqual(mapping[0]["plot_reason"],"missing-cell-or-sample-identity")

    def test_route_not_spectrum_number(self):
        r=fixture("first")
        groups,_=prepare_groups([(Path("r.json"),r)])
        self.assertTrue(groups)
        r["ordinary_rc_eligible"]=False
        groups,mapping=prepare_groups([(Path("r.json"),r)])
        self.assertFalse(groups)
        self.assertIn("not-ordinary",mapping[0]["plot_reason"])

    def test_control_roles_not_mixed(self):
        a,b=fixture("a"),fixture("b",display_role="included")
        groups,_=prepare_groups([(Path("a"),a),(Path("b"),b)])
        self.assertEqual(len(groups),2)

    def test_protocol_not_mixed(self):
        a,b=fixture("a"),fixture("b")
        a["metadata"]["ac_amplitude_mv"]=5
        b["metadata"]["ac_amplitude_mv"]=10
        groups,_=prepare_groups([(Path("a"),a),(Path("b"),b)])
        self.assertEqual(len(groups),2)

    def test_no_silent_unit_mix(self):
        a,b=fixture("a"),fixture("b",impedance_basis="ohm_cm2")
        groups,_=prepare_groups([(Path("a"),a),(Path("b"),b)])
        self.assertEqual(len(groups),2)

    def test_repeats_not_removed(self):
        groups,_=prepare_groups([(Path(str(i)),fixture(str(i),3.1)) for i in range(3)])
        self.assertEqual(len(next(iter(groups.values()))),3)

    def test_common_support_no_interpolation(self):
        a,b=fixture("a"),fixture("b")
        b["curves"]["tau_s"]=b["curves"]["tau_s"][::2]
        b["curves"]["gamma"]=b["curves"]["gamma"][::2]
        groups,_=prepare_groups([(Path("a"),a),(Path("b"),b)])
        curves,lo,hi=clip_group(next(iter(groups.values())))
        for c in curves:
            np.testing.assert_array_equal(c["g"], c["gamma"][(c["tau"] >= 10**lo*(1-1e-14)) &
                                                            (c["tau"] <= 10**hi*(1+1e-14))])
            self.assertGreaterEqual(c["x"].min(),lo)
            self.assertLessEqual(c["x"].max(),hi)

    def test_native_heatmap_edges_bounded(self):
        edges=cell_edges(np.array([1.,2.,3.]),1.,3.)
        np.testing.assert_array_equal(edges,[1,1.5,2.5,3])

    def test_negative_gamma_not_rc_ridge(self):
        r=fixture();r["curves"]["gamma"][10]=-1
        groups,mapping=prepare_groups([(Path("r"),r)])
        self.assertFalse(groups)
        self.assertIn("negative gamma",mapping[0]["plot_reason"])

    def test_numerical_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"results";p.mkdir()
            r=fixture();r["numerical_sha256"]=numerical_digest(r)
            r["curves"]["gamma"][5]+=1
            (p/"r.json").write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,"hash mismatch"):render_trends(Path(d),dpi=72)

    def test_visual_files_all_curves_and_editable_svg(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"results";p.mkdir()
            for i in range(6):
                r=fixture(str(i),3+i*.10)
                (p/f"{i}.json").write_text(json.dumps(r))
            report=render_trends(Path(d),dpi=72,max_curves=4,font="DejaVu Sans")
            self.assertEqual(report["status"],"completed",report["errors"])
            self.assertEqual(len(report["groups"][0]["pages"]),2)
            target=Path(d)/"05_figures/trends/draft"
            with (target/"plot_mapping.csv").open(encoding="utf-8-sig") as h:rows=list(csv.DictReader(h))
            self.assertEqual(sum(r["plotted"]=="True" for r in rows),6)
            for name in report["output_files"]:
                self.assertTrue((target/name).is_file())
                if name.endswith(".svg"):
                    self.assertIn("<text",(target/name).read_text())
            self.assertTrue(all(page[k]["qa"]["pass"] for page in report["groups"][0]["pages"]
                                for k in ("ridge","heatmap")))

    def test_all_twenty_voltage_labels_have_clearance(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"results";p.mkdir()
            for i in range(20):(p/f"{i}.json").write_text(json.dumps(fixture(str(i),3+i*.03)))
            report=render_trends(Path(d),dpi=72,max_curves=20,font="DejaVu Sans")
            self.assertEqual(report["status"],"completed",report["errors"])
            self.assertEqual(report["groups"][0]["spectrum_count"],20)


if __name__=="__main__":
    unittest.main(verbosity=2)

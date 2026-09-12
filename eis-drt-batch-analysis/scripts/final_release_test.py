"""Final release counterexamples for noise-aware routing and source-row exports."""
import copy
import csv
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
from batch_drt import nonseries_inductive_region, source_parser_options
from advanced_smoke_test import rc_branch, rl_branch
from result_contract import export_bundle
from export_branch_identity_test import fixture
from model_recommendation import recommend_model
from result_contract import numerical_digest


class NoiseAwareInductanceTests(unittest.TestCase):
    def test_consumed_series_l_noise_case_does_not_establish_nonseries_response(self):
        f=np.logspace(5.,-3.,61)
        truth=.8+rc_branch(f,3.,30.)+2j*np.pi*f*2e-6
        rng=np.random.default_rng(3723)
        z=truth+.003*abs(truth)*(rng.normal(size=len(f))+1j*rng.normal(size=len(f)))
        before=z.copy()
        old,_=nonseries_inductive_region(f,-z.imag)
        detected,evidence=nonseries_inductive_region(f,-z.imag,impedance_magnitude=abs(z))
        self.assertTrue(old)
        self.assertFalse(detected)
        self.assertEqual(evidence["noise_evidence"]["status"],"estimated")
        self.assertGreater(evidence["noise_limited_negative_points"],0)
        np.testing.assert_array_equal(z,before)

    def test_real_rl_loop_remains_detected_under_development_noise(self):
        f=np.logspace(5.,-3.,61)
        truth=.8+rc_branch(f,3.,30.)+rl_branch(f,1.5,3.)
        for seed in (101,202,303,404):
            rng=np.random.default_rng(seed)
            z=truth+.003*abs(truth)*(rng.normal(size=len(f))+1j*rng.normal(size=len(f)))
            detected,_=nonseries_inductive_region(f,-z.imag,impedance_magnitude=abs(z))
            self.assertTrue(detected,seed)

    def test_screen_is_impedance_scale_and_input_order_covariant(self):
        f=np.logspace(5.,-3.,61)
        z=.8+rc_branch(f,3.,30.)+rl_branch(f,1.5,3.)
        expected,evidence=nonseries_inductive_region(f,-z.imag,impedance_magnitude=abs(z))
        for scale in (1e-6,1e6):
            current,check=nonseries_inductive_region(f[::-1],-z[::-1].imag*scale,impedance_magnitude=abs(z[::-1])*scale)
            self.assertEqual(current,expected)
            np.testing.assert_allclose(check["pointwise_screen_threshold_impedance"],np.array(evidence["pointwise_screen_threshold_impedance"])*scale,rtol=1e-10)

    def test_invalid_modulus_cannot_silently_disable_noise_floor(self):
        with self.assertRaises(ValueError):
            nonseries_inductive_region(np.logspace(4,-2,20),np.ones(20),impedance_magnitude=np.ones(19))


class SourceRowExportTests(unittest.TestCase):
    def test_source_ledger_retains_excluded_line_content(self):
        result,payload=fixture()
        audit={"source":"/synthetic/input.csv","sha256":"a"*64,"row_ledger":[
            {"source_line":1,"raw_line":"Frequency,Zreal,Zimag","status":"header","reason":"declared-header"},
            {"source_line":2,"raw_line":"0,1,-2","status":"excluded","reason":"nonpositive-frequency"},
            {"source_line":3,"raw_line":"=not-a-formula-to-execute","status":"excluded","reason":"nonnumeric"}]}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"execution_contract.json").write_text("{}")
            export_bundle(root,[result],[payload],[],{"source_audits":[audit]}, {})
            with (root/"01_inputs/source_row_ledger.csv").open() as h:rows=list(csv.DictReader(h))
            self.assertEqual([r["source_line"] for r in rows],["1","2","3"])
            self.assertEqual(rows[-1]["raw_line"],audit["row_ledger"][-1]["raw_line"])
            self.assertEqual(json.loads((root/"01_inputs/source_audits.json").read_text()),[audit])

    def test_manifest_order_override_is_explicit_and_validated(self):
        from argparse import Namespace
        args=Namespace(frequency_order="acquisition")
        chosen=source_parser_options(args,{"text_frequency_order":"single-spectrum-descending"})
        self.assertEqual(chosen.frequency_order,"single-spectrum-descending")
        self.assertEqual(args.frequency_order,"acquisition")
        with self.assertRaises(ValueError):source_parser_options(args,{"text_frequency_order":"guess"})


class PredictiveBranchTests(unittest.TestCase):
    def example(self, error=.01, guard="pass"):
        r,p=fixture("predictive_positive_rc")
        c=r["curves"]
        measured=np.array(c["zreal_measured"])-1j*np.array(c["neg_zimag_measured"])
        predicted=measured*(1+error)
        r["signed_gdrt"]={"status":"disabled"}
        r["predictive_rc_competition"]={"status":"completed","selected_candidate":"positive_rc",
            "candidates":{"positive_rc":{"status":"completed","reconstruction_branch":"predictive_positive_rc",
                "tau_s":[.02,.2],"gamma":[7.,8.],"reconstructed":predicted.tolist(),
                "prediction_rms_percent":error*100,"full_refit_guard":{"status":guard},
                "lambda":.001,"lambda_grid":[{"lambda":.001,"prediction_rms_percent":error*100}],
                "support":{"supported_absolute_area_fraction":1.},"reliability_reasons":[]}}}
        return r,p

    def test_cv_selected_branch_export_preserves_primary_and_peak_ownership(self):
        r,p=self.example()
        original=copy.deepcopy(r["curves"])
        r["model_recommendation"]=recommend_model(r)
        self.assertEqual(r["model_recommendation"]["recommended_branch"],"predictive_positive_rc")
        self.assertIn("predictive-individual-peaks-not-validated",r["model_recommendation"]["reasons"])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"execution_contract.json").write_text("{}")
            export_bundle(root,[r],[p],[],{}, {})
            with (root/"03_results/recommended/curves.csv").open() as h:rows=list(csv.DictReader(h))
            self.assertEqual([float(row["gamma"]) for row in rows],[7.,8.])
            self.assertEqual({row["distribution_branch"] for row in rows},{"predictive_positive_rc"})
            self.assertTrue(all(row["selection_is_physical_acceptance"]=="False" for row in rows))
            with (root/"03_results/recommended/reconstruction.csv").open() as h:fit=list(csv.DictReader(h))
            np.testing.assert_allclose([float(row["zreal_reconstructed"]) for row in fit],np.array(original["zreal_measured"])*1.01)
        self.assertEqual(original,r["curves"])

    def test_unviable_or_worse_refit_cannot_replace_primary(self):
        for error,guard in ((2.,"pass"),(.001,"review-required")):
            r,_=self.example(error,guard)
            self.assertEqual(recommend_model(r)["recommended_branch"],"primary_rc")

    def test_predictive_arrays_are_in_checkpoint_digest(self):
        r,_=self.example()
        before=numerical_digest(r)
        r["predictive_rc_competition"]["candidates"]["positive_rc"]["gamma"][0]+=1
        self.assertNotEqual(before,numerical_digest(r))


class SourceJoinTests(unittest.TestCase):
    def test_normalized_rows_have_source_line_join_and_reject_misalignment(self):
        from result_contract import normalized_input_rows
        r,p=fixture()
        p["metadata"]["included_source_lines"]=[8,4,6]
        rows=normalized_input_rows([p],[r])
        self.assertEqual([row["source_line"] for row in rows],[8,4,6])
        self.assertEqual([row["included_point_index"] for row in rows],[1,2,3])
        p["metadata"]["included_source_lines"]=[8,4]
        with self.assertRaises(ValueError):normalized_input_rows([p],[r])

    def test_real_parser_ledger_and_normalized_same_name_index_agree(self):
        from argparse import Namespace
        from eis_parser import read_text
        from result_contract import normalized_input_rows
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/"repeated.csv"
            source.write_text("Frequency (Hz),Zreal (ohm),Zimag (ohm)\n100,2,-1\n10,3,-2\n1,4,-3\n100,5,-1\n10,6,-2\n1,7,-3\n")
            spectra,audit=read_text(source,"joined",Namespace(imag_convention="auto",frequency_unit="auto",impedance_unit="auto"))
            self.assertEqual(len(spectra),2)
            ledger={row["source_line"]:row for row in audit["row_ledger"] if row.get("status")=="included"}
            for spectrum in spectra:
                r,p=fixture()
                p["metadata"]=spectrum.metadata
                p["frequency_hz"]=spectrum.freq_hz
                p["zreal"]=spectrum.zreal
                p["neg_zimag"]=spectrum.neg_zimag
                for row in normalized_input_rows([p],[r]):
                    self.assertEqual(row["included_point_index"],ledger[row["source_line"]]["included_point_index"])


if __name__=="__main__":unittest.main()

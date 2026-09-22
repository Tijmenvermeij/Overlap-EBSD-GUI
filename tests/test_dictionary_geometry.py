import json
import unittest
from dataclasses import replace, asdict
from multistep_overlap_ebsd.phases import DictionaryProvenance, DictionaryAsset


class DictionaryGeometryTests(unittest.TestCase):
    def test_explicit_acceptance_is_scoped_to_pc_and_tilt(self):
        saved = dict(pc_bruker=[.48,.34,.86], detector_tilt=1.9, sample_tilt=70., software_binning=3)
        original = DictionaryProvenance.create('master', {'phase':1}, saved)
        for changes in ({'detector_tilt':1.25}, {'pc_bruker':[.48,.17,.78]},
                        {'detector_tilt':1.25,'pc_bruker':[.48,.17,.78]}):
            with self.subTest(changes=changes):
                settings = dict(saved, **changes)
                current = replace(original, settings_json=json.dumps(settings))
                geometry = {k:settings[k] for k in ('pc_bruker','detector_tilt')}
                self.assertTrue(original.reusable_geometry_difference(current))
                self.assertFalse(DictionaryAsset(provenance=original).compatible_with(current))
                asset = DictionaryAsset(provenance=original, accepted_geometry=geometry)
                self.assertTrue(asset.compatible_with(current))
                restored = json.loads(json.dumps(asdict(asset)))
                restored['provenance'] = DictionaryProvenance(**restored['provenance'])
                self.assertTrue(DictionaryAsset(**restored).compatible_with(current))
                for extra in ({'detector_tilt':2.}, {'pc_bruker':[.5,.5,.5]},
                              {'sample_tilt':71.}, {'software_binning':2}):
                    self.assertFalse(asset.compatible_with(replace(current, settings_json=json.dumps(dict(settings, **extra)))))
                self.assertFalse(asset.compatible_with(replace(current, master_sha256='other')))
                self.assertFalse(asset.compatible_with(replace(current, structure_sha256='other')))

    def test_gui_requires_confirmation_for_geometry_reuse(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from multistep_overlap_ebsd.gui_phases import PhaseControls
        saved = dict(pc_bruker=[.5,.5,.6], detector_tilt=1.9)
        current = dict(pc_bruker=[.5,.5,.6], detector_tilt=1.25)
        for accepted in (False, True):
            session = Mock()
            session.dictionary_geometry_difference.return_value = (saved, current)
            gui = SimpleNamespace(_selected_phase_key=lambda:'phase', session=session,
                                  _run_threaded=Mock(), _refresh_phase_table=Mock())
            with patch('multistep_overlap_ebsd.gui_phases.filedialog.askopenfilename', return_value='dictionary.h5'), patch(
                    'multistep_overlap_ebsd.gui_phases.messagebox.askyesno', return_value=accepted) as warning:
                PhaseControls._load_phase_dictionary(gui)
            self.assertIn('1.9°', warning.call_args.args[1])
            self.assertIn('1.25°', warning.call_args.args[1])
            if accepted:
                gui._run_threaded.call_args.args[0]()
                session.load_phase_dictionary.assert_called_once_with('phase', 'dictionary.h5', accepted_geometry=current)
            else:
                gui._run_threaded.assert_not_called()

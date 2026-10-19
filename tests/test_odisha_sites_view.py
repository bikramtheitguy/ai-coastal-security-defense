"""Exercise source filtering and exports without a Streamlit server."""
import importlib.util
import io
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class OdishaViewTests(unittest.TestCase):
    def run_view(self, state, coast, district='All source districts'):
        ui = Mock()
        ui.expander.return_value = nullcontext()
        ui.selectbox.return_value = district
        downloads = ModuleType('dashboard.downloads')
        downloads.download_button = Mock()
        with patch.dict(sys.modules, {'streamlit': ui, 'dashboard.downloads': downloads}):
            spec = importlib.util.spec_from_file_location(
                'odisha_view_under_test', ROOT / 'dashboard/odisha_sites_view.py'
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.render_odisha_sites(state, coast)
        return ui, downloads.download_button

    def test_odisha_exports_all_source_entries(self):
        ui, download = self.run_view('IN-OR', 'East coast')
        frame = ui.dataframe.call_args.args[0]
        self.assertEqual(len(frame), 25)
        exported = pd.read_csv(io.BytesIO(download.call_args.args[1]))
        self.assertEqual(exported.district.nunique(), 7)
        self.assertTrue(exported.latitude.isna().all())
        self.assertTrue(exported.longitude.isna().all())
        self.assertEqual(exported.operational_status_source.notna().sum(), 1)
        self.assertEqual(exported.iloc[22].site_name_source, 'Gopalpur FH (Defunct)')

    def test_district_filters_display_and_download(self):
        ui, download = self.run_view('All coastal states/UTs', 'All coasts', 'Ganjam')
        self.assertEqual(len(ui.dataframe.call_args.args[0]), 5)
        exported = pd.read_csv(io.BytesIO(download.call_args.args[1]))
        self.assertEqual(set(exported.district), {'Ganjam'})

    def test_other_state_has_no_records_or_download(self):
        ui, download = self.run_view('IN-GA', 'All coasts')
        ui.dataframe.assert_not_called()
        download.assert_not_called()

    def test_west_coast_has_no_odisha_records(self):
        ui, download = self.run_view('All coastal states/UTs', 'West coast')
        ui.dataframe.assert_not_called()
        download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
